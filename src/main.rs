mod app;
mod inference;
mod tools;
mod ui;

use std::{
    env, fs, io,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{LlamaChatTemplate, LlamaModel, params::LlamaModelParams},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::Position,
};

use app::{App, LogLevel, Mode, ModelInfo, StatusKind, Tab};
use inference::{Generation, StepResult};

fn main() -> Result<()> {
    install_panic_hook();

    let model_path = env::var_os("MODEL_PATH")
        .map(PathBuf::from)
        .context("falta MODEL_PATH; indica la ruta a un archivo .gguf")?;
    if !model_path.is_file() {
        bail!(
            "el modelo no existe o no es un archivo: {}",
            model_path.display()
        );
    }

    // llama.cpp escribe sus logs en stderr y se pintan encima de la TUI.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let backend = LlamaBackend::init().context("no se pudo inicializar llama.cpp")?;
    // Por defecto todas las capas a la GPU; N_GPU_LAYERS=0 fuerza CPU.
    let gpu_layers = env::var("N_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(999);
    let params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers);
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)
        .with_context(|| format!("no se pudo cargar {}", model_path.display()))?;
    let template = model
        .chat_template(None)
        .context("el modelo no contiene una plantilla de chat")?;

    let n_ctx = env::var("N_CTX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 256)
        .unwrap_or_else(|| (model.n_ctx_train() as usize).min(16384).max(512));
    let max_tokens = env::var("MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256);
    let temperature = env::var("TEMPERATURE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.35)
        .max(0.0);
    let top_p = env::var("TOP_P")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.9)
        .clamp(0.01, 1.0);

    let info = ModelInfo::new(&model_path, &model, n_ctx, max_tokens, temperature, top_p);
    let mut app = App::load(&model, &template, info)?;
    let result = run_tui(&mut app, &model, &backend, &template);
    restore_terminal();

    let save_result = app.save();
    if let Err(err) = &save_result {
        eprintln!("no se pudieron guardar las sesiones: {err:#}");
    }
    result?;
    save_result?;
    Ok(())
}

fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        default_hook(info);
    }));
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
}

fn run_tui<'m>(
    app: &mut App,
    model: &'m LlamaModel,
    backend: &LlamaBackend,
    template: &LlamaChatTemplate,
) -> Result<()> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let mut generation: Option<Generation<'m>> = None;
    let mut quit = false;

    while !quit {
        let mut finished = false;
        let mut failure = None;
        if let Some(running) = generation.as_mut() {
            let started = Instant::now();
            loop {
                match running.step(model) {
                    Ok(StepResult::Continue) => {
                        if started.elapsed() >= Duration::from_millis(18) {
                            break;
                        }
                    }
                    Ok(StepResult::Done) => {
                        finished = true;
                        break;
                    }
                    Err(err) => {
                        failure = Some(err);
                        break;
                    }
                }
            }
        }
        if let Some(err) = failure {
            generation = None;
            app.log(
                LogLevel::Error,
                format!("Generación abortada: {err:#}"),
            );
            app.set_status(StatusKind::Error, "La generación falló; revisa Debug");
            app.mark_context_dirty();
        }
        if finished {
            if let Some(running) = generation.take() {
                let outcome = running.outcome();
                app.finish_run(outcome, model, template);
            }
        }
        if generation.is_none() {
            if let Some(index) = app.continue_agent.take() {
                start_generation(app, model, backend, template, &mut generation, Some(index));
            }
        }

        terminal.draw(|frame| ui::draw(frame, app, model, template, generation.as_ref()))?;

        let timeout = if generation.is_some() {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(120)
        };
        if !event::poll(timeout)? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                quit = handle_key(app, key, &mut generation, model, backend, template)?;
            }
            Event::Mouse(mouse) => {
                handle_mouse(app, mouse, &mut generation, model);
            }
            _ => {}
        }
    }
    Ok(())
}

fn handle_key<'m>(
    app: &mut App,
    key: KeyEvent,
    generation: &mut Option<Generation<'m>>,
    model: &'m LlamaModel,
    backend: &LlamaBackend,
    template: &LlamaChatTemplate,
) -> Result<bool> {
    if !app.mode.is_normal() {
        return handle_modal_key(app, key, generation, model);
    }

    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('q')) => return Ok(true),
        (KeyModifiers::CONTROL, KeyCode::Char('n')) => {
            app.new_session(None, model);
        }
        (KeyModifiers::CONTROL, KeyCode::Char('p')) => app.previous_session(model),
        (KeyModifiers::CONTROL, KeyCode::Char('l')) => app.next_session(model),
        (KeyModifiers::CONTROL, KeyCode::Char('s')) => match app.save() {
            Ok(()) => {
                app.log(LogLevel::Ok, "Sesiones guardadas manualmente".into());
                app.set_status(StatusKind::Ok, "Sesiones guardadas");
            }
            Err(err) => report(app, "No se pudo guardar", err),
        },
        (KeyModifiers::CONTROL, KeyCode::Char('f')) => app.open_add_file(),
        (KeyModifiers::CONTROL, KeyCode::Char('y')) => app.approve_run(),
        (KeyModifiers::CONTROL, KeyCode::Char('g')) => {
            app.show_debug = !app.show_debug;
            app.set_status(
                StatusKind::Info,
                if app.show_debug {
                    "Panel de depuración visible · Ctrl-G lo oculta"
                } else {
                    "Panel de depuración oculto · Ctrl-G lo muestra"
                },
            );
        }
        (KeyModifiers::CONTROL, KeyCode::Char('r')) => app.open_rename(),
        (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
            if generation.is_some() {
                app.set_status(
                    StatusKind::Warn,
                    "No puedes borrar sesiones mientras se genera",
                );
            } else {
                app.open_delete();
            }
        }
        (_, KeyCode::F(1)) => app.set_tab(Tab::Chat),
        (_, KeyCode::F(2)) => app.set_tab(Tab::Context),
        (_, KeyCode::F(3)) => app.set_tab(Tab::Debug),
        (_, KeyCode::F(4)) => app.set_tab(Tab::Help),
        (_, KeyCode::BackTab) => app.set_tab(app.tab.previous()),
        (_, KeyCode::Tab) => {
            app.set_tab(if shift {
                app.tab.previous()
            } else {
                app.tab.next()
            });
        }
        (_, KeyCode::Esc) => {
            if generation.is_some() {
                cancel_generation(app, generation);
            } else if app.pending_run.is_some() {
                app.reject_run();
            } else if !app.input.is_empty() {
                app.input.clear();
                app.cursor = 0;
                app.mark_context_dirty();
                app.set_status(StatusKind::Info, "Entrada borrada");
            }
        }
        (_, KeyCode::Enter) => {
            if app.tab == Tab::Chat {
                return submit_input(app, generation, model, backend, template);
            }
        }
        (_, KeyCode::Up) => scroll_by(app, true, 1),
        (_, KeyCode::Down) => scroll_by(app, false, 1),
        (_, KeyCode::PageUp) => scroll_by(app, true, 8),
        (_, KeyCode::PageDown) => scroll_by(app, false, 8),
        (_, KeyCode::Home) if app.tab != Tab::Chat => jump_scroll(app, false),
        (_, KeyCode::End) if app.tab != Tab::Chat => jump_scroll(app, true),
        _ => {}
    }

    if app.tab == Tab::Chat && edit_line(&mut app.input, &mut app.cursor, &key) {
        app.mark_context_dirty();
    }
    Ok(false)
}

fn handle_modal_key<'m>(
    app: &mut App,
    key: KeyEvent,
    generation: &mut Option<Generation<'m>>,
    model: &LlamaModel,
) -> Result<bool> {
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('q'))
    {
        return Ok(true);
    }

    if matches!(key.code, KeyCode::Esc) {
        app.close_modal();
        return Ok(false);
    }

    if matches!(key.code, KeyCode::Enter) {
        match app.mode.clone() {
            Mode::AddFile { .. } | Mode::RenameSession { .. } => submit_modal(app, model),
            Mode::ConfirmDelete { index, selected } => {
                if selected == 0 {
                    apply_delete(app, index, generation);
                } else {
                    app.close_modal();
                }
            }
            Mode::FilePreview { .. } => app.close_modal(),
            Mode::Normal => {}
        }
        return Ok(false);
    }

    if let Mode::ConfirmDelete { index, selected } = app.mode.clone() {
        match key.code {
            KeyCode::Left | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => {
                if let Mode::ConfirmDelete {
                    selected: value, ..
                } = &mut app.mode
                {
                    *value = 1 - selected;
                }
            }
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                apply_delete(app, index, generation);
            }
            KeyCode::Char('n') | KeyCode::Char('N') => app.close_modal(),
            _ => {}
        }
        return Ok(false);
    }

    if matches!(app.mode, Mode::FilePreview { .. })
        && matches!(key.code, KeyCode::Char('q'))
    {
        app.close_modal();
        return Ok(false);
    }

    if let Mode::AddFile { input, .. } = &app.mode {
        let delta: isize = match key.code {
            KeyCode::Up => -1,
            KeyCode::Down => 1,
            KeyCode::PageUp => -8,
            KeyCode::PageDown => 8,
            _ => 0,
        };
        if delta != 0 {
            let count = app.file_matches(input).len();
            move_picker(app, delta, count);
            return Ok(false);
        }
    }

    match &mut app.mode {
        Mode::AddFile {
            input,
            cursor,
            selected,
            error,
        } => {
            let before = input.clone();
            edit_line(input, cursor, &key);
            if *input != before {
                *selected = 0;
                *error = None;
            }
        }
        Mode::RenameSession { input, cursor, .. } => {
            edit_line(input, cursor, &key);
        }
        Mode::FilePreview { scroll, .. } => match key.code {
            KeyCode::Up => *scroll = scroll.saturating_sub(1),
            KeyCode::Down => *scroll = scroll.saturating_add(1),
            KeyCode::PageUp => *scroll = scroll.saturating_sub(8),
            KeyCode::PageDown => *scroll = scroll.saturating_add(8),
            KeyCode::Home => *scroll = 0,
            KeyCode::End => *scroll = u16::MAX,
            _ => {}
        },
        _ => {}
    }
    Ok(false)
}

fn submit_input<'m>(
    app: &mut App,
    generation: &mut Option<Generation<'m>>,
    model: &'m LlamaModel,
    backend: &LlamaBackend,
    template: &LlamaChatTemplate,
) -> Result<bool> {
    let prompt = app.input.trim().to_owned();
    if prompt.is_empty() {
        return Ok(false);
    }
    if generation.is_some() {
        app.set_status(
            StatusKind::Warn,
            "Hay una generación en curso; pulsa Esc para cancelarla",
        );
        return Ok(false);
    }
    match handle_command(app, &prompt, model, generation) {
        CommandResult::NotCommand => {
            // Un mensaje nuevo con un comando pendiente cuenta como rechazo.
            app.reject_run();
            start_generation(app, model, backend, template, generation, None);
            Ok(false)
        }
        CommandResult::Handled => Ok(false),
        CommandResult::Quit => Ok(true),
    }
}

enum CommandResult {
    NotCommand,
    Handled,
    Quit,
}

fn handle_command(
    app: &mut App,
    prompt: &str,
    model: &LlamaModel,
    generation: &mut Option<Generation<'_>>,
) -> CommandResult {
    let Some(stripped) = prompt.strip_prefix('/') else {
        return CommandResult::NotCommand;
    };
    let (command, rest) = match stripped.split_once(char::is_whitespace) {
        Some((command, rest)) => (command, rest.trim()),
        None => (stripped, ""),
    };
    match command {
        "salir" | "exit" => CommandResult::Quit,
        "archivo" => {
            clear_input(app);
            if rest.is_empty() {
                app.open_add_file();
            } else {
                match app.add_file(rest, model) {
                    Ok(()) => {}
                    Err(err) => report(app, "No se pudo añadir el archivo", err),
                }
            }
            CommandResult::Handled
        }
        "quitar-archivo" => {
            clear_input(app);
            app.remove_last_file();
            CommandResult::Handled
        }
        "nueva" => {
            clear_input(app);
            let name = if rest.is_empty() { None } else { Some(rest.to_owned()) };
            app.new_session(name, model);
            CommandResult::Handled
        }
        "renombrar" => {
            clear_input(app);
            if rest.is_empty() {
                app.open_rename();
            } else {
                match app.rename_session(app.active, rest) {
                    Ok(()) => {}
                    Err(err) => report(app, "No se pudo renombrar", err),
                }
            }
            CommandResult::Handled
        }
        "borrar" => {
            clear_input(app);
            if generation.is_some() {
                app.set_status(
                    StatusKind::Warn,
                    "No puedes borrar sesiones mientras se genera",
                );
            } else {
                app.open_delete();
            }
            CommandResult::Handled
        }
        "guardar" => {
            clear_input(app);
            match app.save() {
                Ok(()) => {
                    app.log(LogLevel::Ok, "Sesiones guardadas manualmente".into());
                    app.set_status(StatusKind::Ok, "Sesiones guardadas");
                }
                Err(err) => report(app, "No se pudo guardar", err),
            }
            CommandResult::Handled
        }
        "limpiar" => {
            clear_input(app);
            app.logs.clear();
            app.debug_offset = 0;
            app.set_status(StatusKind::Ok, "Registro de depuración limpiado");
            CommandResult::Handled
        }
        "contexto" => {
            clear_input(app);
            app.set_tab(Tab::Context);
            app.mark_context_dirty();
            CommandResult::Handled
        }
        "debug" => {
            clear_input(app);
            app.set_tab(Tab::Debug);
            CommandResult::Handled
        }
        "ayuda" => {
            clear_input(app);
            app.set_tab(Tab::Help);
            CommandResult::Handled
        }
        "chat" => {
            clear_input(app);
            app.set_tab(Tab::Chat);
            CommandResult::Handled
        }
        "cancelar" => {
            clear_input(app);
            cancel_generation(app, generation);
            CommandResult::Handled
        }
        _ => {
            clear_input(app);
            app.set_status(
                StatusKind::Warn,
                format!("Comando desconocido: /{command} · F4 para ver la ayuda"),
            );
            CommandResult::Handled
        }
    }
}

fn start_generation<'m>(
    app: &mut App,
    model: &'m LlamaModel,
    backend: &LlamaBackend,
    template: &LlamaChatTemplate,
    generation: &mut Option<Generation<'m>>,
    continuation: Option<usize>,
) {
    let (session_index, prompt) = match continuation {
        Some(index) => (index.min(app.sessions.len().saturating_sub(1)), String::new()),
        None => (app.active, app.input.trim().to_owned()),
    };
    if continuation.is_none() {
        if prompt.is_empty() {
            return;
        }
        app.agent_steps = 0;
    }
    // Al continuar no se reenvían los archivos; el modelo puede usar read_file.
    let (name, selected) = {
        let session = &app.sessions[session_index];
        let selected = if continuation.is_some() { Vec::new() } else { session.selected_files.clone() };
        (session.name.clone(), selected)
    };
    let mut files = Vec::with_capacity(selected.len());
    for path in &selected {
        match fs::read_to_string(path) {
            Ok(content) => files.push((path.clone(), content)),
            Err(err) => {
                let message = format!("No se pudo leer {path}: {err}");
                app.log(LogLevel::Error, message.clone());
                app.set_status(StatusKind::Error, message);
                return;
            }
        }
    }
    let user_content = continuation
        .is_none()
        .then(|| app::compose_user_content(&prompt, &files).0);
    let messages = match app::chat_messages(&app.sessions[session_index], user_content, tools::uses_xml(template)) {
        Ok(messages) => messages,
        Err(err) => {
            report(app, "No se pudo preparar el mensaje", err);
            return;
        }
    };

    let (formatted, tokens) = match app::format_prompt(model, template, &messages) {
        Ok(value) => value,
        Err(err) => {
            report(app, "No se pudo aplicar la plantilla de chat", err);
            return;
        }
    };
    if tokens.len() + 8 >= app.model.n_ctx {
        let message = format!(
            "El contexto ocupa {} de {} tokens. Acorta la conversación, quita archivos o crea otra sesión (Ctrl-N).",
            tokens.len(),
            app.model.n_ctx
        );
        app.log(LogLevel::Error, message.clone());
        app.set_status(StatusKind::Error, message);
        return;
    }
    let max_new = app
        .model
        .max_tokens
        .min(app.model.n_ctx.saturating_sub(tokens.len() + 1));
    if max_new == 0 {
        app.set_status(
            StatusKind::Error,
            "No queda espacio para generar; acorta el contexto",
        );
        return;
    }

    for (path, content) in &files {
        app.update_file_cache(path, content.clone(), model);
    }

    app.log(
        LogLevel::Gen,
        format!(
            "Prompt a {name}: {} tokens de contexto · {} archivos · máx {} nuevos",
            tokens.len(),
            selected.len(),
            max_new
        ),
    );
    app.log(
        LogLevel::Gen,
        format!(
            "Ajustes: n_ctx {} · temp {:.2} · top_p {:.2}",
            app.model.n_ctx, app.model.temperature, app.model.top_p
        ),
    );

    match Generation::new(
        session_index,
        prompt,
        formatted,
        tokens,
        app.model.n_ctx,
        max_new,
        app.model.temperature,
        app.model.top_p,
        backend,
        model,
    ) {
        Ok(running) => {
            *generation = Some(running);
            if continuation.is_none() {
                app.input.clear();
                app.cursor = 0;
            }
            app.chat_offset = 0;
            app.mark_context_dirty();
            app.set_status(StatusKind::Info, format!("Generando en {name}…"));
        }
        Err(err) => report(app, "No se pudo iniciar la generación", err),
    }
}

fn cancel_generation(app: &mut App, generation: &mut Option<Generation<'_>>) {
    let Some(running) = generation.take() else {
        app.set_status(StatusKind::Info, "No hay ninguna generación en curso");
        return;
    };
    if app.input.is_empty() {
        app.input = running.prompt.clone();
        app.cursor = app.input.chars().count();
    }
    app.log(
        LogLevel::Warn,
        format!(
            "Generación cancelada tras {} tokens; el mensaje volvió al campo",
            running.produced
        ),
    );
    app.set_status(
        StatusKind::Warn,
        "Generación cancelada; el mensaje volvió al campo",
    );
    app.mark_context_dirty();
}

fn handle_mouse<'m>(
    app: &mut App,
    mouse: MouseEvent,
    generation: &mut Option<Generation<'m>>,
    model: &'m LlamaModel,
) {
    app.mouse = Some((mouse.column, mouse.row));
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            handle_click(app, mouse.column, mouse.row, generation, model);
        }
        MouseEventKind::ScrollUp => handle_scroll(app, mouse.column, mouse.row, true),
        MouseEventKind::ScrollDown => handle_scroll(app, mouse.column, mouse.row, false),
        _ => {}
    }
}

fn handle_click<'m>(
    app: &mut App,
    x: u16,
    y: u16,
    generation: &mut Option<Generation<'m>>,
    model: &LlamaModel,
) {
    let point = Position::new(x, y);

    if !app.mode.is_normal() {
        let Some(layout) = ui::modal_layout(app.screen, &app.mode) else {
            return;
        };
        match app.mode.clone() {
            Mode::AddFile { input, .. } if layout.body.contains(point) => {
                let start = ui::picker_start(
                    match app.mode {
                        Mode::AddFile { selected, .. } => selected,
                        _ => 0,
                    },
                    layout.body.height as usize,
                );
                let index = start + (y - layout.body.y) as usize;
                if let Some(path) = app.file_matches(&input).get(index).cloned() {
                    submit_add_file(app, model, &path);
                }
            }
            Mode::AddFile { .. } | Mode::RenameSession { .. } => {
                if layout.actions[0].contains(point) {
                    submit_modal(app, model);
                } else if layout.actions[1].contains(point) || !layout.area.contains(point) {
                    app.close_modal();
                }
            }
            Mode::ConfirmDelete { index, .. } => {
                if layout.actions[0].contains(point) {
                    apply_delete(app, index, generation);
                } else if layout.actions[1].contains(point) {
                    app.close_modal();
                }
            }
            Mode::FilePreview { .. } => {
                if layout.actions[0].contains(point) || !layout.area.contains(point) {
                    app.close_modal();
                }
            }
            Mode::Normal => {}
        }
        return;
    }

    let layout = ui::layout(app.screen, app.tab, app.show_debug);
    if layout.header.contains(point) {
        if let Some(tab) = ui::tab_at(layout.header, x, y) {
            app.set_tab(tab);
        }
        return;
    }
    if let Some(debug_rect) = layout.debug {
        if debug_rect.contains(point) {
            app.set_tab(Tab::Debug);
            app.debug_offset = 0;
            return;
        }
    }

    match layout.areas {
        ui::Areas::Chat {
            sessions,
            chat,
            files,
        } => {
            if let Some(rect) = sessions {
                let parts = ui::sidebar_rects(rect);
                if parts.new.contains(point) {
                    app.new_session(None, model);
                    return;
                }
                if parts.rename.contains(point) {
                    app.open_rename();
                    return;
                }
                if parts.delete.contains(point) {
                    if generation.is_some() {
                        app.set_status(
                            StatusKind::Warn,
                            "No puedes borrar sesiones mientras se genera",
                        );
                    } else {
                        app.open_delete();
                    }
                    return;
                }
                if parts.list.contains(point) {
                    let row = y.saturating_sub(parts.list.y) as usize;
                    let index = app.session_offset as usize + row;
                    app.switch_session(index, model);
                    return;
                }
            }
            if let Some(rect) = files {
                let parts = ui::files_rects(rect);
                if parts.add.contains(point) {
                    app.open_add_file();
                    return;
                }
                if parts.list.contains(point) {
                    let row = y.saturating_sub(parts.list.y) as usize;
                    let index = app.file_offset as usize + row;
                    if index < app.current().selected_files.len() {
                        if x < parts.list.x + 3 {
                            app.remove_file(index);
                        } else {
                            let path = app.current().selected_files[index].clone();
                            app.open_preview(&path);
                        }
                    }
                    return;
                }
            }
            let (history, _) = ui::chat_parts(chat);
            if history.contains(point) {
                app.mouse = Some((x, y));
            }
        }
        ui::Areas::Context { .. } => {}
        ui::Areas::Debug { toolbar, .. } => {
            let parts = ui::debug_toolbar_rects(toolbar);
            if parts.clear.contains(point) {
                app.logs.clear();
                app.debug_offset = 0;
                app.set_status(StatusKind::Ok, "Registro de depuración limpiado");
                return;
            }
            if parts.follow.contains(point) {
                app.debug_offset = 0;
                app.set_status(StatusKind::Info, "Debug siguiendo el final");
                return;
            }
        }
        ui::Areas::Help { .. } => {}
    }
}

fn submit_modal(app: &mut App, model: &LlamaModel) {
    match app.mode.clone() {
        Mode::AddFile {
            input, selected, ..
        } => {
            let path = app.file_matches(&input).get(selected).cloned();
            submit_add_file(app, model, path.as_deref().unwrap_or(&input));
        }
        Mode::RenameSession { input, .. } => submit_rename(app, &input),
        _ => {}
    }
}

fn move_picker(app: &mut App, delta: isize, count: usize) {
    if let Mode::AddFile { selected, .. } = &mut app.mode {
        let max = count.saturating_sub(1) as isize;
        *selected = (*selected as isize + delta).clamp(0, max) as usize;
    }
}

fn submit_add_file(app: &mut App, model: &LlamaModel, input: &str) {
    let path = input.trim();
    if path.is_empty() {
        set_modal_error(app, "Escribe una ruta relativa");
        return;
    }
    match app.add_file(path, model) {
        Ok(()) => app.close_modal(),
        Err(err) => {
            let message = format!("{err:#}");
            set_modal_error(app, &message);
            app.set_status(StatusKind::Error, message);
        }
    }
}

fn submit_rename(app: &mut App, input: &str) {
    match app.rename_session(app.active, input) {
        Ok(()) => app.close_modal(),
        Err(err) => {
            let message = format!("{err:#}");
            set_modal_error(app, &message);
            app.set_status(StatusKind::Error, message);
        }
    }
}

fn set_modal_error(app: &mut App, message: &str) {
    match &mut app.mode {
        Mode::AddFile { error, .. } | Mode::RenameSession { error, .. } => {
            *error = Some(message.to_owned());
        }
        _ => {}
    }
}

fn apply_delete<'m>(
    app: &mut App,
    index: usize,
    generation: &mut Option<Generation<'m>>,
) {
    if generation.is_some() {
        app.close_modal();
        app.set_status(
            StatusKind::Warn,
            "No puedes borrar sesiones mientras se genera",
        );
        return;
    }
    app.close_modal();
    if let Err(err) = app.delete_session(index) {
        report(app, "No se pudo borrar la sesión", err);
    }
}

fn handle_scroll(app: &mut App, x: u16, y: u16, up: bool) {
    let point = Position::new(x, y);
    if !app.mode.is_normal() {
        if let Mode::AddFile { input, .. } = &app.mode {
            let count = app.file_matches(input).len();
            move_picker(app, if up { -1 } else { 1 }, count);
            return;
        }
        if let Mode::FilePreview { scroll, .. } = &mut app.mode {
            *scroll = if up {
                scroll.saturating_sub(3)
            } else {
                scroll.saturating_add(3)
            };
        }
        return;
    }

    let layout = ui::layout(app.screen, app.tab, app.show_debug);
    if let Some(debug_rect) = layout.debug {
        if debug_rect.contains(point) {
            scroll_debug(app, up, 3);
            return;
        }
    }
    match layout.areas {
        ui::Areas::Chat {
            sessions,
            chat,
            files,
        } => {
            if files.map(|rect| rect.contains(point)).unwrap_or(false) {
                app.file_offset = if up {
                    app.file_offset.saturating_sub(1)
                } else {
                    app.file_offset.saturating_add(1)
                };
            } else if sessions.map(|rect| rect.contains(point)).unwrap_or(false) {
                app.session_offset = if up {
                    app.session_offset.saturating_sub(1)
                } else {
                    app.session_offset.saturating_add(1)
                };
            } else {
                let (history, _) = ui::chat_parts(chat);
                if history.contains(point) {
                    app.chat_offset = if up {
                        app.chat_offset.saturating_add(3)
                    } else {
                        app.chat_offset.saturating_sub(3)
                    };
                }
            }
        }
        ui::Areas::Context { text, .. } => {
            if text.contains(point) {
                app.context_scroll = if up {
                    app.context_scroll.saturating_sub(3)
                } else {
                    app.context_scroll.saturating_add(3)
                };
            }
        }
        ui::Areas::Debug { log, .. } => {
            if log.contains(point) {
                scroll_debug(app, up, 3);
            }
        }
        ui::Areas::Help { text } => {
            if text.contains(point) {
                app.help_scroll = if up {
                    app.help_scroll.saturating_sub(3)
                } else {
                    app.help_scroll.saturating_add(3)
                };
            }
        }
    }
}

fn scroll_by(app: &mut App, up: bool, amount: u16) {
    match app.tab {
        Tab::Chat => {
            app.chat_offset = if up {
                app.chat_offset.saturating_add(amount)
            } else {
                app.chat_offset.saturating_sub(amount)
            };
        }
        Tab::Context => {
            app.context_scroll = if up {
                app.context_scroll.saturating_sub(amount)
            } else {
                app.context_scroll.saturating_add(amount)
            };
        }
        Tab::Debug => scroll_debug(app, up, amount),
        Tab::Help => {
            app.help_scroll = if up {
                app.help_scroll.saturating_sub(amount)
            } else {
                app.help_scroll.saturating_add(amount)
            };
        }
    }
}

fn scroll_debug(app: &mut App, up: bool, amount: u16) {
    app.debug_offset = if up {
        app.debug_offset.saturating_add(amount)
    } else {
        app.debug_offset.saturating_sub(amount)
    };
}

fn jump_scroll(app: &mut App, bottom: bool) {
    match app.tab {
        Tab::Chat => app.chat_offset = if bottom { 0 } else { u16::MAX },
        Tab::Context => {
            app.context_scroll = if bottom { u16::MAX } else { 0 };
        }
        Tab::Debug => {
            app.debug_offset = if bottom { 0 } else { u16::MAX };
        }
        Tab::Help => {
            app.help_scroll = if bottom { u16::MAX } else { 0 };
        }
    }
}

fn clear_input(app: &mut App) {
    app.input.clear();
    app.cursor = 0;
    app.mark_context_dirty();
}

fn report(app: &mut App, label: &str, err: anyhow::Error) {
    app.log(LogLevel::Error, format!("{label}: {err:#}"));
    app.set_status(StatusKind::Error, format!("{label}: {err}"));
}

fn edit_line(input: &mut String, cursor: &mut usize, key: &KeyEvent) -> bool {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Left => {
            *cursor = cursor.saturating_sub(1);
            true
        }
        KeyCode::Right => {
            *cursor = (*cursor + 1).min(input.chars().count());
            true
        }
        KeyCode::Home => {
            *cursor = 0;
            true
        }
        KeyCode::End => {
            *cursor = input.chars().count();
            true
        }
        KeyCode::Backspace => {
            if *cursor > 0 {
                let start = char_to_byte(input, *cursor - 1);
                let end = char_to_byte(input, *cursor);
                input.replace_range(start..end, "");
                *cursor -= 1;
            }
            true
        }
        KeyCode::Delete => {
            if *cursor < input.chars().count() {
                let start = char_to_byte(input, *cursor);
                let end = char_to_byte(input, *cursor + 1);
                input.replace_range(start..end, "");
            }
            true
        }
        KeyCode::Char('a') if ctrl => {
            *cursor = 0;
            true
        }
        KeyCode::Char('e') if ctrl => {
            *cursor = input.chars().count();
            true
        }
        KeyCode::Char('u') if ctrl => {
            input.clear();
            *cursor = 0;
            true
        }
        KeyCode::Char('w') if ctrl => {
            delete_previous_word(input, cursor);
            true
        }
        KeyCode::Char(ch) if insertable(key.modifiers) => {
            let index = char_to_byte(input, *cursor);
            input.insert(index, ch);
            *cursor += 1;
            true
        }
        _ => false,
    }
}

fn insertable(modifiers: KeyModifiers) -> bool {
    if modifiers.contains(KeyModifiers::ALT) && !modifiers.contains(KeyModifiers::CONTROL) {
        return false;
    }
    !modifiers.contains(KeyModifiers::CONTROL) || modifiers.contains(KeyModifiers::ALT)
}

fn delete_previous_word(input: &mut String, cursor: &mut usize) {
    let chars: Vec<char> = input.chars().collect();
    let mut start = (*cursor).min(chars.len());
    while start > 0 && chars[start - 1].is_whitespace() {
        start -= 1;
    }
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let byte_start = char_to_byte(input, start);
    let byte_end = char_to_byte(input, *cursor);
    if byte_start < byte_end {
        input.replace_range(byte_start..byte_end, "");
        *cursor = start;
    }
}

fn char_to_byte(value: &str, char_index: usize) -> usize {
    value
        .char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(value.len())
}
