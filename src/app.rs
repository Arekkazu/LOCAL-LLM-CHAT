use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use llama_cpp_2::{
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use ratatui::layout::Rect;
use serde::{Deserialize, Serialize};

use crate::inference::{RunOutcome, RunStats};
use crate::tools::{self, ToolCall};

const SESSION_DIR: &str = ".llm-cheap/sessions";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Context,
    Debug,
    Help,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Chat, Tab::Context, Tab::Debug, Tab::Help];

    pub fn title(self) -> &'static str {
        match self {
            Tab::Chat => "Chat",
            Tab::Context => "Contexto",
            Tab::Debug => "Debug",
            Tab::Help => "Ayuda",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Tab::Chat => " Chat ",
            Tab::Context => " Contexto ",
            Tab::Debug => " Debug ",
            Tab::Help => " Ayuda ",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Tab::Chat => Tab::Context,
            Tab::Context => Tab::Debug,
            Tab::Debug => Tab::Help,
            Tab::Help => Tab::Chat,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Tab::Chat => Tab::Help,
            Tab::Context => Tab::Chat,
            Tab::Debug => Tab::Context,
            Tab::Help => Tab::Debug,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    Info,
    Ok,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Ok,
    Warn,
    Error,
    Gen,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Ok => "OK",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERR",
            LogLevel::Gen => "GEN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub at: String,
    pub level: LogLevel,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedSession {
    pub name: String,
    pub messages: Vec<Message>,
    pub selected_files: Vec<String>,
    /// Directorio de trabajo del agente, relativo a la raíz del proyecto.
    #[serde(default = "root_dir")]
    pub cwd: String,
}

fn root_dir() -> String {
    ".".into()
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: String,
    pub content: String,
    pub tokens: usize,
    pub bytes: usize,
    pub error: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ContextPreview {
    pub text: String,
    pub total_tokens: usize,
    pub system_tokens: usize,
    pub history_tokens: usize,
    pub files_tokens: usize,
    pub input_tokens: usize,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: String,
    pub name: String,
    pub bytes: u64,
    pub n_ctx_train: usize,
    pub n_ctx: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl ModelInfo {
    pub fn new(
        path: &Path,
        model: &LlamaModel,
        n_ctx: usize,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Self {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("modelo.gguf")
            .to_owned();
        let bytes = fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        Self {
            path: path.display().to_string(),
            name,
            bytes,
            n_ctx_train: model.n_ctx_train() as usize,
            n_ctx,
            max_tokens,
            temperature,
            top_p,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Mode {
    Normal,
    AddFile {
        input: String,
        cursor: usize,
        error: Option<String>,
        selected: usize,
    },
    RenameSession {
        input: String,
        cursor: usize,
        error: Option<String>,
    },
    ConfirmDelete {
        index: usize,
        selected: usize,
    },
    FilePreview {
        path: String,
        content: String,
        tokens: usize,
        bytes: usize,
        scroll: u16,
        error: bool,
    },
}

impl Mode {
    pub fn is_normal(&self) -> bool {
        matches!(self, Mode::Normal)
    }
}

pub struct App {
    pub sessions: Vec<SavedSession>,
    pub active: usize,
    pub tab: Tab,
    pub input: String,
    pub cursor: usize,
    pub status: String,
    pub status_kind: StatusKind,
    pub mode: Mode,
    pub logs: Vec<LogEntry>,
    pub started: Instant,
    pub screen: Rect,
    pub mouse: Option<(u16, u16)>,
    pub session_offset: u16,
    pub file_offset: u16,
    pub chat_offset: u16,
    pub context_scroll: u16,
    pub debug_offset: u16,
    pub help_scroll: u16,
    pub context: ContextPreview,
    pub context_dirty: bool,
    pub file_cache: HashMap<String, FileInfo>,
    pub model: ModelInfo,
    pub last_stats: Option<RunStats>,
    pub show_debug: bool,
    pub project_files: Vec<String>,
    /// Comando que el modelo pidió ejecutar: (sesión, comando). Ctrl-Y aprueba, Esc rechaza.
    pub pending_run: Option<(usize, String)>,
    /// Sesión cuya generación debe continuar tras un resultado de herramienta.
    pub continue_agent: Option<usize>,
    pub agent_steps: usize,
}

impl App {
    pub fn load(model: &LlamaModel, template: &LlamaChatTemplate, info: ModelInfo) -> Result<Self> {
        let directory = Path::new(SESSION_DIR);
        fs::create_dir_all(directory).context("no se pudo crear .llm-cheap/sessions")?;

        let mut sessions = Vec::new();
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let data = fs::read_to_string(&path)?;
            match serde_json::from_str::<SavedSession>(&data) {
                Ok(session) => sessions.push(session),
                Err(err) => eprintln!("sesión ignorada ({}): {err}", path.display()),
            }
        }
        sessions.sort_by(|left, right| left.name.cmp(&right.name));
        if sessions.is_empty() {
            sessions.push(SavedSession {
                name: "sesion-1".into(),
                messages: Vec::new(),
                selected_files: Vec::new(),
                cwd: root_dir(),
            });
        }

        let mut app = Self {
            sessions,
            active: 0,
            tab: Tab::Chat,
            input: String::new(),
            cursor: 0,
            status: "Listo · ratón activo · Tab o F1-F4 cambian de pestaña".into(),
            status_kind: StatusKind::Info,
            mode: Mode::Normal,
            logs: Vec::new(),
            started: Instant::now(),
            screen: Rect::new(0, 0, 80, 24),
            mouse: None,
            session_offset: 0,
            file_offset: 0,
            chat_offset: 0,
            context_scroll: 0,
            debug_offset: 0,
            help_scroll: 0,
            context: ContextPreview::default(),
            context_dirty: true,
            file_cache: HashMap::new(),
            model: info,
            last_stats: None,
            show_debug: false,
            project_files: Vec::new(),
            pending_run: None,
            continue_agent: None,
            agent_steps: 0,
        };

        app.log(
            LogLevel::Info,
            format!(
                "Modelo: {} · {:.1} MB · n_ctx {} (entrenado para {})",
                app.model.name,
                app.model.bytes as f64 / 1_048_576.0,
                app.model.n_ctx,
                app.model.n_ctx_train
            ),
        );
        app.log(
            LogLevel::Info,
            "Plantilla de chat del modelo cargada correctamente".into(),
        );
        app.log(
            LogLevel::Info,
            format!(
                "Sesiones cargadas: {} · límite de generación: {} tokens",
                app.sessions.len(),
                app.model.max_tokens
            ),
        );
        app.log(
            LogLevel::Info,
            "Ratón: clic en pestañas, sesiones, archivos y botones; rueda para desplazar".into(),
        );

        app.reload_files(model, true, true);
        app.refresh_context(model, template);
        Ok(app)
    }

    pub fn current(&self) -> &SavedSession {
        &self.sessions[self.active]
    }

    pub fn current_mut(&mut self) -> &mut SavedSession {
        &mut self.sessions[self.active]
    }

    pub fn log(&mut self, level: LogLevel, message: String) {
        let secs = self.started.elapsed().as_secs();
        let at = format!("{:02}:{:02}", secs / 60, secs % 60);
        self.logs.push(LogEntry { at, level, message });
        if self.logs.len() > 900 {
            let excess = self.logs.len() - 900;
            self.logs.drain(0..excess);
        } else if self.debug_offset > 0 {
            self.debug_offset = self.debug_offset.saturating_add(1);
        }
    }

    pub fn set_status(&mut self, kind: StatusKind, text: impl Into<String>) {
        self.status_kind = kind;
        self.status = text.into();
    }

    pub fn set_tab(&mut self, tab: Tab) {
        if self.tab != tab {
            self.tab = tab;
            if tab == Tab::Context {
                self.context_dirty = true;
            }
        }
    }

    pub fn mark_context_dirty(&mut self) {
        self.context_dirty = true;
    }

    pub fn open_add_file(&mut self) {
        self.project_files = list_project_files();
        self.mode = Mode::AddFile {
            input: String::new(),
            cursor: 0,
            error: None,
            selected: 0,
        };
    }

    /// Archivos del proyecto cuyo path contiene todas las palabras de `query`.
    pub fn file_matches(&self, query: &str) -> Vec<String> {
        let terms = query
            .split_whitespace()
            .map(str::to_lowercase)
            .collect::<Vec<_>>();
        self.project_files
            .iter()
            .filter(|path| {
                let lower = path.to_lowercase();
                terms.iter().all(|term| lower.contains(term))
            })
            .cloned()
            .collect()
    }

    pub fn open_rename(&mut self) {
        let name = self.current().name.clone();
        self.mode = Mode::RenameSession {
            input: name.clone(),
            cursor: name.chars().count(),
            error: None,
        };
    }

    pub fn open_delete(&mut self) {
        if self.sessions.len() <= 1 {
            self.set_status(
                StatusKind::Warn,
                "No puedes borrar la única sesión; crea otra con Ctrl-N",
            );
            return;
        }
        self.mode = Mode::ConfirmDelete {
            index: self.active,
            selected: 1,
        };
    }

    pub fn open_preview(&mut self, path: &str) {
        let Some(info) = self.file_cache.get(path) else {
            self.set_status(StatusKind::Warn, format!("El archivo {path} no está cargado"));
            return;
        };
        self.mode = Mode::FilePreview {
            path: path.to_owned(),
            content: info.content.clone(),
            tokens: info.tokens,
            bytes: info.bytes,
            scroll: 0,
            error: info.error,
        };
    }

    pub fn close_modal(&mut self) {
        self.mode = Mode::Normal;
    }

    pub fn save(&self) -> Result<()> {
        let directory = Path::new(SESSION_DIR);
        fs::create_dir_all(directory).context("no se pudo crear .llm-cheap/sessions")?;

        let mut used = HashSet::new();
        let mut written = HashSet::new();
        for session in &self.sessions {
            let base = sanitize_name(&session.name);
            let mut stem = base.clone();
            let mut suffix = 2;
            while used.contains(&stem) {
                stem = format!("{base}-{suffix}");
                suffix += 1;
            }
            used.insert(stem.clone());
            let file = format!("{stem}.json");
            let data = serde_json::to_string_pretty(session)?;
            fs::write(directory.join(&file), data)?;
            written.insert(file);
        }

        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Some(file) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if written.contains(file) {
                continue;
            }
            let Ok(data) = fs::read_to_string(&path) else {
                continue;
            };
            if serde_json::from_str::<SavedSession>(&data).is_ok() {
                let _ = fs::remove_file(path);
            }
        }
        Ok(())
    }

    pub fn autosave(&mut self) {
        if let Err(err) = self.save() {
            self.log(
                LogLevel::Error,
                format!("No se pudieron guardar las sesiones: {err:#}"),
            );
        }
    }

    pub fn new_session(&mut self, requested: Option<String>, model: &LlamaModel) -> String {
        let base = requested
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("sesion-{}", self.sessions.len() + 1));
        let name = self.unique_name(&base, None);
        self.sessions.push(SavedSession {
            name: name.clone(),
            messages: Vec::new(),
            selected_files: Vec::new(),
            cwd: root_dir(),
        });
        self.active = self.sessions.len() - 1;
        self.reset_views();
        self.file_offset = 0;
        self.context_dirty = true;
        self.reload_files(model, false, false);
        self.log(LogLevel::Ok, format!("Sesión creada: {name}"));
        self.set_status(StatusKind::Ok, format!("Sesión nueva: {name}"));
        self.autosave();
        name
    }

    pub fn switch_session(&mut self, index: usize, model: &LlamaModel) {
        if index >= self.sessions.len() || index == self.active {
            return;
        }
        self.active = index;
        self.reset_views();
        self.context_dirty = true;
        self.reload_files(model, false, false);
        let name = self.current().name.clone();
        self.log(LogLevel::Info, format!("Sesión activa: {name}"));
        self.set_status(StatusKind::Info, format!("Sesión: {name}"));
    }

    pub fn next_session(&mut self, model: &LlamaModel) {
        let index = (self.active + 1) % self.sessions.len();
        self.switch_session(index, model);
    }

    pub fn previous_session(&mut self, model: &LlamaModel) {
        let index = if self.active == 0 {
            self.sessions.len() - 1
        } else {
            self.active - 1
        };
        self.switch_session(index, model);
    }

    fn reset_views(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.chat_offset = 0;
        self.context_scroll = 0;
        self.file_offset = 0;
    }

    pub fn rename_session(&mut self, index: usize, requested: &str) -> Result<()> {
        let name = requested.trim();
        if name.is_empty() {
            bail!("el nombre no puede estar vacío");
        }
        let name = truncate_chars(name, 60);
        if self.name_conflicts(&name, Some(index)) {
            bail!("ya existe una sesión con ese nombre");
        }
        let Some(session) = self.sessions.get_mut(index) else {
            bail!("la sesión ya no existe");
        };
        let old = std::mem::replace(&mut session.name, name.clone());
        self.log(
            LogLevel::Ok,
            format!("Sesión renombrada: {old} → {name}"),
        );
        self.set_status(StatusKind::Ok, format!("Sesión renombrada a {name}"));
        self.autosave();
        Ok(())
    }

    pub fn delete_session(&mut self, index: usize) -> Result<()> {
        if self.sessions.len() <= 1 {
            bail!("no puedes borrar la única sesión");
        }
        if index >= self.sessions.len() {
            bail!("la sesión ya no existe");
        }
        let removed = self.sessions.remove(index);
        if self.active >= self.sessions.len() {
            self.active = self.sessions.len() - 1;
        } else if index < self.active {
            self.active -= 1;
        }
        self.reset_views();
        self.context_dirty = true;
        self.log(
            LogLevel::Warn,
            format!("Sesión eliminada: {} ({} mensajes)", removed.name, removed.messages.len()),
        );
        self.set_status(StatusKind::Warn, format!("Sesión eliminada: {}", removed.name));
        self.autosave();
        Ok(())
    }

    pub fn add_file(&mut self, path: &str, model: &LlamaModel) -> Result<()> {
        let relative = validate_project_path(path)?;
        if !relative.is_file() {
            bail!("el archivo no existe: {}", relative.display());
        }
        let value = relative.to_string_lossy().to_string();
        if self.current().selected_files.iter().any(|item| item == &value) {
            bail!("el archivo ya está seleccionado: {value}");
        }
        let content = fs::read_to_string(&relative)
            .with_context(|| format!("no se pudo leer {value} (¿es un archivo binario?)"))?;
        let tokens = count_tokens(model, &content);
        let bytes = content.as_bytes().len();
        self.current_mut().selected_files.push(value.clone());
        self.file_cache.insert(
            value.clone(),
            FileInfo {
                path: value.clone(),
                content,
                tokens,
                bytes,
                error: false,
            },
        );
        self.context_dirty = true;
        self.log(
            LogLevel::Ok,
            format!("Archivo añadido: {value} · {} B · {} tokens", bytes, tokens),
        );
        self.set_status(StatusKind::Ok, format!("Archivo añadido: {value}"));
        self.autosave();
        Ok(())
    }

    pub fn remove_file(&mut self, index: usize) {
        let removed = {
            let files = &mut self.current_mut().selected_files;
            if index >= files.len() {
                return;
            }
            files.remove(index)
        };
        self.file_cache.remove(&removed);
        self.context_dirty = true;
        self.log(LogLevel::Warn, format!("Archivo quitado: {removed}"));
        self.set_status(StatusKind::Warn, format!("Archivo quitado: {removed}"));
        self.autosave();
    }

    pub fn remove_last_file(&mut self) {
        let index = self.current().selected_files.len().saturating_sub(1);
        if self.current().selected_files.is_empty() {
            self.set_status(StatusKind::Warn, "No hay archivos seleccionados");
        } else {
            self.remove_file(index);
        }
    }

    pub fn reload_files(&mut self, model: &LlamaModel, force: bool, announce: bool) {
        let selected = self.current().selected_files.clone();
        for path in selected {
            if !force && self.file_cache.contains_key(&path) {
                continue;
            }
            let info = read_file_info(&path, model);
            if info.error && announce {
                self.log(
                    LogLevel::Warn,
                    format!("No se pudo leer {path}; revisa la ruta o los permisos"),
                );
            }
            self.file_cache.insert(path, info);
        }
        self.context_dirty = true;
    }

    pub fn update_file_cache(&mut self, path: &str, content: String, model: &LlamaModel) {
        let tokens = count_tokens(model, &content);
        let bytes = content.as_bytes().len();
        self.file_cache.insert(
            path.to_owned(),
            FileInfo {
                path: path.to_owned(),
                content,
                tokens,
                bytes,
                error: false,
            },
        );
    }

    pub fn refresh_context(&mut self, model: &LlamaModel, template: &LlamaChatTemplate) {
        let session = &self.sessions[self.active];
        let files = session
            .selected_files
            .iter()
            .map(|path| {
                let content = self
                    .file_cache
                    .get(path)
                    .map(|info| info.content.clone())
                    .unwrap_or_else(|| format!("[archivo no cargado: {path}]"));
                (path.clone(), content)
            })
            .collect::<Vec<_>>();
        let (user_content, files_block) = compose_user_content(&self.input, &files);

        let mut messages = Vec::with_capacity(session.messages.len() + 2);
        let system = tools::system_prompt(&session.cwd, tools::uses_xml(template));
        match LlamaChatMessage::new("system".into(), system.clone()) {
            Ok(message) => messages.push(message),
            Err(err) => {
                self.context = ContextPreview {
                    text: format!("[error] no se pudo crear el mensaje de sistema: {err}"),
                    note: Some("Hay un carácter nulo en el prompt de sistema".into()),
                    ..ContextPreview::default()
                };
                self.context_dirty = false;
                return;
            }
        }
        let mut history_tokens = 0;
        for message in &session.messages {
            history_tokens += count_tokens(model, &message.content);
            match LlamaChatMessage::new(message.role.clone(), message.content.clone()) {
                Ok(value) => messages.push(value),
                Err(err) => {
                    self.context = ContextPreview {
                        text: format!("[error] mensaje inválido en el historial: {err}"),
                        note: Some("El historial contiene un carácter nulo".into()),
                        ..ContextPreview::default()
                    };
                    self.context_dirty = false;
                    return;
                }
            }
        }
        let input_tokens = count_tokens(model, &self.input);
        let files_tokens = count_tokens(model, &files_block);
        let system_tokens = count_tokens(model, &system);
        match LlamaChatMessage::new("user".into(), user_content) {
            Ok(user) => messages.push(user),
            Err(err) => {
                self.context = ContextPreview {
                    text: format!("[error] no se pudo preparar el mensaje: {err}"),
                    note: Some("La entrada contiene un carácter nulo".into()),
                    ..ContextPreview::default()
                };
                self.context_dirty = false;
                return;
            }
        }

        match format_prompt(model, template, &messages) {
            Ok((text, tokens)) => {
                let mut note = None;
                if tokens.len() >= self.model.n_ctx {
                    note = Some(format!(
                        "No cabe en el contexto: {} tokens de {}. Acorta la conversación, quita archivos o crea otra sesión.",
                        tokens.len(),
                        self.model.n_ctx
                    ));
                }
                self.context = ContextPreview {
                    text,
                    total_tokens: tokens.len(),
                    system_tokens,
                    history_tokens,
                    files_tokens,
                    input_tokens,
                    note,
                };
            }
            Err(err) => {
                self.context = ContextPreview {
                    text: format!("[error] no se pudo aplicar la plantilla de chat:\n{err:#}"),
                    system_tokens,
                    history_tokens,
                    files_tokens,
                    input_tokens,
                    note: Some("La plantilla del modelo rechazó el contexto".into()),
                    ..ContextPreview::default()
                };
            }
        }
        self.context_dirty = false;
    }

    pub fn finish_run(
        &mut self,
        outcome: RunOutcome,
        model: &LlamaModel,
        template: &LlamaChatTemplate,
    ) {
        let index = outcome.session.min(self.sessions.len().saturating_sub(1));
        let session_name = self.sessions[index].name.clone();
        if !outcome.prompt.is_empty() {
            self.sessions[index].messages.push(Message {
                role: "user".into(),
                content: outcome.prompt.clone(),
            });
        }
        self.sessions[index].messages.push(Message {
            role: "assistant".into(),
            content: outcome.response.clone(),
        });

        let stats = outcome.stats.clone();
        self.log(
            LogLevel::Gen,
            format!(
                "Respuesta en {session_name}: {} tokens en {} ms ({:.1} t/s) · prompt {} tokens en {} ms",
                stats.generated, stats.generation_ms, stats.tok_per_s, stats.prompt_tokens, stats.prefill_ms
            ),
        );

        if let Some(call) = tools::parse_tool_call(&outcome.response) {
            self.handle_tool_call(index, call);
        } else {
            self.set_status(
                StatusKind::Ok,
                format!(
                    "Respuesta lista en {session_name} · {} tokens · {:.1} t/s",
                    stats.generated, stats.tok_per_s
                ),
            );
        }

        self.last_stats = Some(stats);
        self.chat_offset = 0;
        self.context_dirty = true;
        // Forzado: write_file pudo cambiar un archivo seleccionado.
        self.reload_files(model, true, false);
        self.refresh_context(model, template);
        self.autosave();
    }

    fn handle_tool_call(&mut self, index: usize, call: Result<ToolCall, String>) {
        self.agent_steps += 1;
        if self.agent_steps > tools::MAX_STEPS {
            self.set_status(
                StatusKind::Warn,
                format!("Límite de {} pasos de herramienta; escribe para continuar", tools::MAX_STEPS),
            );
            return;
        }
        let call = match call {
            Ok(call) => call,
            Err(err) => {
                self.log(LogLevel::Warn, format!("Llamada a herramienta inválida: {err}"));
                return self.push_tool_result(index, format!("error: {err}"), true);
            }
        };
        self.log(LogLevel::Info, format!("Herramienta: {} {}", call.name, call.args));
        if call.name == "run" {
            let command = call.arg("command").unwrap_or_default().to_owned();
            self.set_status(
                StatusKind::Warn,
                format!("¿Ejecutar `{command}`? Ctrl-Y ejecuta · Esc rechaza"),
            );
            self.pending_run = Some((index, command));
            return;
        }
        let result = match tools::execute(&call, &mut self.sessions[index].cwd) {
            Ok(text) => text,
            Err(err) => format!("error: {err:#}"),
        };
        self.set_status(StatusKind::Info, format!("{} ejecutado; continuando…", call.name));
        self.push_tool_result(index, result, true);
    }

    pub fn approve_run(&mut self) {
        let Some((index, command)) = self.pending_run.take() else {
            return;
        };
        let index = index.min(self.sessions.len().saturating_sub(1));
        self.log(LogLevel::Info, format!("Ejecutando: {command}"));
        let output = tools::run(&self.sessions[index].cwd, &command);
        self.set_status(StatusKind::Info, "Comando ejecutado; continuando…");
        self.push_tool_result(index, output, true);
    }

    pub fn reject_run(&mut self) {
        let Some((index, command)) = self.pending_run.take() else {
            return;
        };
        let index = index.min(self.sessions.len().saturating_sub(1));
        self.log(LogLevel::Warn, format!("Comando rechazado: {command}"));
        self.set_status(StatusKind::Warn, "Comando rechazado");
        self.push_tool_result(index, "El usuario rechazó el comando.".into(), false);
    }

    fn push_tool_result(&mut self, index: usize, text: String, resume: bool) {
        self.sessions[index].messages.push(Message {
            role: "user".into(),
            content: tools::response(&text),
        });
        if resume {
            self.continue_agent = Some(index);
        }
        self.context_dirty = true;
        self.autosave();
    }

    fn unique_name(&self, base: &str, skip: Option<usize>) -> String {
        let base = truncate_chars(base.trim(), 60);
        let base = if base.is_empty() {
            "sesion".to_owned()
        } else {
            base
        };
        if !self.name_conflicts(&base, skip) {
            return base;
        }
        for suffix in 2..10_000 {
            let candidate = format!("{base}-{suffix}");
            if !self.name_conflicts(&candidate, skip) {
                return candidate;
            }
        }
        format!("{base}-{}", self.sessions.len() + 1)
    }

    fn name_conflicts(&self, name: &str, skip: Option<usize>) -> bool {
        let stem = sanitize_name(name);
        self.sessions.iter().enumerate().any(|(index, session)| {
            if Some(index) == skip {
                return false;
            }
            session.name.eq_ignore_ascii_case(name) || sanitize_name(&session.name) == stem
        })
    }
}

pub fn sanitize_name(name: &str) -> String {
    let mut value = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            value.push(ch.to_ascii_lowercase());
        } else if ch.is_whitespace() {
            value.push('-');
        }
    }
    let value = value.trim_matches('-').to_owned();
    if value.is_empty() {
        "sesion".into()
    } else {
        value
    }
}

pub fn truncate_chars(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_owned();
    }
    let mut result = value.chars().take(max.saturating_sub(1)).collect::<String>();
    result.push('…');
    result
}

pub fn count_tokens(model: &LlamaModel, text: &str) -> usize {
    model.vocab().tokenize(text.as_bytes(), false, false).len()
}

pub fn compose_user_content(prompt: &str, files: &[(String, String)]) -> (String, String) {
    let mut content = prompt.to_owned();
    let mut block = String::new();
    if !files.is_empty() {
        block.push_str("\n\nARCHIVOS SELECCIONADOS:\n");
        for (path, body) in files {
            block.push_str(&format!("\n--- {path} ---\n{body}\n"));
        }
    }
    content.push_str(&block);
    (content, block)
}

/// `user_content` es `None` al continuar tras un resultado de herramienta.
pub fn chat_messages(
    session: &SavedSession,
    user_content: Option<String>,
    xml_tools: bool,
) -> Result<Vec<LlamaChatMessage>> {
    let mut messages = Vec::with_capacity(session.messages.len() + 2);
    messages.push(
        LlamaChatMessage::new(
            "system".into(),
            tools::system_prompt(&session.cwd, xml_tools),
        )
            .context("no se pudo crear el mensaje de sistema")?,
    );
    for message in &session.messages {
        messages.push(
            LlamaChatMessage::new(message.role.clone(), message.content.clone())
                .context("el historial contiene un carácter nulo")?,
        );
    }
    if let Some(content) = user_content {
        messages.push(
            LlamaChatMessage::new("user".into(), content)
                .context("el mensaje contiene un carácter nulo")?,
        );
    }
    Ok(messages)
}

pub fn format_prompt(
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    messages: &[LlamaChatMessage],
) -> Result<(String, Vec<LlamaToken>)> {
    let mut formatted = model
        .apply_chat_template(template, messages, true)
        .context("no se pudo aplicar la plantilla de chat")?;
    // La jinja de Qwen3/3.5 desactiva el razonamiento con un bloque <think> vacío;
    // la plantilla clásica de llama.cpp no lo añade y el modelo gastaría MAX_TOKENS pensando.
    if template.to_str().is_ok_and(|text| text.contains("enable_thinking")) {
        formatted.push_str("<think>\n\n</think>\n\n");
    }
    let tokens = model.vocab().tokenize(formatted.as_bytes(), true, true);
    Ok((formatted, tokens))
}

fn read_file_info(path: &str, model: &LlamaModel) -> FileInfo {
    match fs::read_to_string(path) {
        Ok(content) => {
            let tokens = count_tokens(model, &content);
            let bytes = content.as_bytes().len();
            FileInfo {
                path: path.to_owned(),
                content,
                tokens,
                bytes,
                error: false,
            }
        }
        Err(err) => FileInfo {
            path: path.to_owned(),
            content: format!("[no se pudo leer {path}: {err}]"),
            tokens: 0,
            bytes: 0,
            error: true,
        },
    }
}

// ponytail: recorrido completo al abrir el diálogo, tope de 5000 archivos; usar `ignore` si el proyecto crece
fn list_project_files() -> Vec<String> {
    const SKIP: [&str; 4] = ["target", "models", "node_modules", "result"];
    let mut found = Vec::new();
    let mut pending = vec![PathBuf::from(".")];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || SKIP.contains(&name.as_str()) {
                continue;
            }
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => pending.push(path),
                Ok(kind) if kind.is_file() => {
                    let value = path.strip_prefix(".").unwrap_or(&path);
                    found.push(value.to_string_lossy().to_string());
                }
                _ => {}
            }
            if found.len() >= 5000 {
                break;
            }
        }
    }
    found.sort();
    found
}

pub fn validate_project_path(path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    if candidate.is_absolute() || path.contains("..") {
        bail!("solo se permiten rutas relativas dentro del proyecto");
    }
    let root = env::current_dir()?.canonicalize()?;
    let full = root.join(candidate);
    let canonical = full.canonicalize().context("no se pudo validar la ruta")?;
    if !canonical.starts_with(&root) {
        bail!("ruta fuera del proyecto");
    }
    Ok(canonical.strip_prefix(&root)?.to_path_buf())
}
