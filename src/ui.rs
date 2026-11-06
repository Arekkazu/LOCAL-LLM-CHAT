use std::time::Duration;

use llama_cpp_2::{model::LlamaChatTemplate, model::LlamaModel};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Margin, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Gauge, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthChar;

use crate::{
    app::{App, LogLevel, Mode, StatusKind, Tab, truncate_chars},
    inference::Generation,
};

const ACCENT: Color = Color::Cyan;
const USER: Color = Color::Cyan;
const ASSISTANT: Color = Color::Green;
const MUTED: Color = Color::DarkGray;
const OK: Color = Color::Green;
const WARN: Color = Color::Yellow;
const ERROR: Color = Color::Red;
// Colores del medidor de contexto: uno por parte del prompt.
const SEG_SYSTEM: Color = Color::Blue;
const SEG_HISTORY: Color = Color::Cyan;
const SEG_FILES: Color = Color::Magenta;
const SEG_INPUT: Color = Color::Yellow;
const SEG_REPLY: Color = Color::Green;
const BRAND: &str = " LLM·CHEAP ";

const BTN_ADD: &str = "[ Añadir ]";
const BTN_CANCEL: &str = "[ Cancelar ]";
const BTN_RENAME: &str = "[ Renombrar ]";
const BTN_DELETE: &str = "[ Eliminar ]";
const BTN_CLOSE: &str = "[ Cerrar ]";
const BTN_CLEAR: &str = "[ Limpiar ]";
const BTN_FOLLOW: &str = "[ Seguir ]";

#[derive(Clone, Copy)]
pub struct UiLayout {
    pub header: Rect,
    pub body: Rect,
    pub debug: Option<Rect>,
    pub footer: Rect,
    pub areas: Areas,
}

#[derive(Clone, Copy)]
pub enum Areas {
    Chat {
        sessions: Option<Rect>,
        chat: Rect,
        files: Option<Rect>,
    },
    Context {
        stats: Rect,
        text: Rect,
    },
    Debug {
        toolbar: Rect,
        log: Rect,
    },
    Help {
        text: Rect,
    },
}

pub struct SidebarRects {
    pub new: Rect,
    pub list: Rect,
    pub rename: Rect,
    pub delete: Rect,
}

pub struct FilesRects {
    pub add: Rect,
    pub list: Rect,
    pub footer: Rect,
}

pub struct DebugToolbarRects {
    pub clear: Rect,
    pub follow: Rect,
}

pub struct ModalLayout {
    pub area: Rect,
    pub instruction: Rect,
    pub input: Option<Rect>,
    pub feedback: Option<Rect>,
    pub body: Rect,
    pub actions: [Rect; 2],
    pub labels: [&'static str; 2],
}

pub fn layout(area: Rect, tab: Tab, show_debug: bool) -> UiLayout {
    let header_h = area.height.min(1);
    let footer_h = if area.height >= 12 {
        2
    } else if area.height >= 7 {
        1
    } else {
        0
    };
    let header = Rect::new(area.x, area.y, area.width, header_h);
    let footer = Rect::new(
        area.x,
        area.bottom().saturating_sub(footer_h),
        area.width,
        footer_h,
    );
    let middle_y = area.y + header_h;
    let middle_h = footer.y.saturating_sub(middle_y);
    let debug_h = if tab == Tab::Chat && show_debug {
        if middle_h >= 22 {
            7
        } else if middle_h >= 17 {
            5
        } else {
            0
        }
    } else {
        0
    };
    let body_h = middle_h.saturating_sub(debug_h);
    let body = Rect::new(area.x, middle_y, area.width, body_h);
    let debug = if debug_h > 0 {
        Some(Rect::new(area.x, middle_y + body_h, area.width, debug_h))
    } else {
        None
    };
    let areas = match tab {
        Tab::Chat => chat_areas(body),
        Tab::Context => context_areas(body),
        Tab::Debug => debug_areas(body),
        Tab::Help => Areas::Help { text: body },
    };
    UiLayout {
        header,
        body,
        debug,
        footer,
        areas,
    }
}

fn chat_areas(body: Rect) -> Areas {
    let width = body.width;
    let mut sessions_w = if width >= 110 {
        26
    } else if width >= 88 {
        22
    } else if width >= 64 {
        18
    } else {
        0
    };
    let mut files_w = if width >= 110 {
        34
    } else if width >= 88 {
        28
    } else if width >= 72 {
        22
    } else {
        0
    };
    if width.saturating_sub(sessions_w + files_w) < 24 {
        if files_w > 0 && width.saturating_sub(sessions_w) >= 24 {
            files_w = 0;
        } else {
            sessions_w = 0;
            files_w = 0;
        }
    }
    let chat_w = width.saturating_sub(sessions_w + files_w);
    let sessions = if sessions_w > 0 {
        Some(Rect::new(body.x, body.y, sessions_w, body.height))
    } else {
        None
    };
    let chat = Rect::new(body.x + sessions_w, body.y, chat_w, body.height);
    let files = if files_w > 0 {
        Some(Rect::new(
            body.x + sessions_w + chat_w,
            body.y,
            files_w,
            body.height,
        ))
    } else {
        None
    };
    Areas::Chat {
        sessions,
        chat,
        files,
    }
}

fn context_areas(body: Rect) -> Areas {
    if body.width >= 84 {
        let stats_w = ((body.width as u32 * 40 / 100) as u16)
            .clamp(34, 52)
            .min(body.width.saturating_sub(30));
        let stats = Rect::new(body.x, body.y, stats_w, body.height);
        let text = Rect::new(
            body.x + stats_w,
            body.y,
            body.width.saturating_sub(stats_w),
            body.height,
        );
        Areas::Context { stats, text }
    } else {
        let stats_h = body.height.saturating_sub(4).min(12);
        let stats = Rect::new(body.x, body.y, body.width, stats_h);
        let text = Rect::new(
            body.x,
            body.y + stats_h,
            body.width,
            body.height.saturating_sub(stats_h),
        );
        Areas::Context { stats, text }
    }
}

fn debug_areas(body: Rect) -> Areas {
    let toolbar_h = body.height.min(1);
    let toolbar = Rect::new(body.x, body.y, body.width, toolbar_h);
    let log = Rect::new(
        body.x,
        body.y + toolbar_h,
        body.width,
        body.height.saturating_sub(toolbar_h),
    );
    Areas::Debug { toolbar, log }
}

pub fn chat_parts(rect: Rect) -> (Rect, Rect) {
    let input_h = rect.height.min(3);
    let history_h = rect.height.saturating_sub(input_h);
    let history = Rect::new(rect.x, rect.y, rect.width, history_h);
    let input = Rect::new(rect.x, rect.y + history_h, rect.width, input_h);
    (history, input)
}

pub fn sidebar_rects(rect: Rect) -> SidebarRects {
    let inner = rect.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width == 0 || inner.height == 0 {
        let zero = Rect::new(inner.x, inner.y, 0, 0);
        return SidebarRects {
            new: zero,
            list: zero,
            rename: zero,
            delete: zero,
        };
    }
    let new = Rect::new(inner.x, inner.y, inner.width, 1);
    let list = Rect::new(
        inner.x,
        inner.y + 1,
        inner.width,
        inner.height.saturating_sub(2),
    );
    let footer_y = inner.y + inner.height.saturating_sub(1);
    let half = inner.width / 2;
    SidebarRects {
        new,
        list,
        rename: Rect::new(inner.x, footer_y, half, 1),
        delete: Rect::new(inner.x + half, footer_y, inner.width.saturating_sub(half), 1),
    }
}

pub fn files_rects(rect: Rect) -> FilesRects {
    let inner = rect.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width == 0 || inner.height == 0 {
        let zero = Rect::new(inner.x, inner.y, 0, 0);
        return FilesRects {
            add: zero,
            list: zero,
            footer: zero,
        };
    }
    FilesRects {
        add: Rect::new(inner.x, inner.y, inner.width, 1),
        list: Rect::new(
            inner.x,
            inner.y + 1,
            inner.width,
            inner.height.saturating_sub(2),
        ),
        footer: Rect::new(
            inner.x,
            inner.y + inner.height.saturating_sub(1),
            inner.width,
            1,
        ),
    }
}

pub fn debug_toolbar_rects(rect: Rect) -> DebugToolbarRects {
    let w1 = (BTN_CLEAR.chars().count() + 2) as u16;
    let w2 = (BTN_FOLLOW.chars().count() + 2) as u16;
    DebugToolbarRects {
        clear: Rect::new(rect.x, rect.y, w1, 1),
        follow: Rect::new(rect.x + w1 + 1, rect.y, w2, 1),
    }
}

fn brand_width(area: Rect) -> u16 {
    if area.width >= 78 {
        BRAND.chars().count() as u16 + 1
    } else {
        0
    }
}

pub fn tab_rects(area: Rect) -> Vec<(Tab, Rect)> {
    let mut x = area.x + brand_width(area);
    let mut rects = Vec::new();
    for (index, tab) in Tab::ALL.iter().enumerate() {
        let width = tab.label().chars().count() as u16;
        rects.push((*tab, Rect::new(x, area.y, width, 1)));
        x = x.saturating_add(width);
        if index + 1 < Tab::ALL.len() {
            x = x.saturating_add(1);
        }
    }
    rects
}

pub fn tab_at(area: Rect, x: u16, y: u16) -> Option<Tab> {
    let point = Position::new(x, y);
    tab_rects(area)
        .into_iter()
        .find(|(_, rect)| rect.contains(point))
        .map(|(tab, _)| tab)
}

pub fn modal_layout(screen: Rect, mode: &Mode) -> Option<ModalLayout> {
    let (area, is_preview) = match mode {
        Mode::Normal => return None,
        Mode::AddFile { .. } => (
            centered(screen, 70, ((screen.height as u32 * 70 / 100) as u16).max(12)),
            false,
        ),
        Mode::RenameSession { .. } => (centered(screen, 62, 9), false),
        Mode::ConfirmDelete { .. } => (centered(screen, 58, 8), false),
        Mode::FilePreview { .. } => (
            centered(
                screen,
                86,
                ((screen.height as u32 * 84 / 100) as u16).max(8).min(screen.height),
            ),
            true,
        ),
    };
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width == 0 || inner.height == 0 {
        return None;
    }
    let labels = match mode {
        Mode::AddFile { .. } => [BTN_ADD, BTN_CANCEL],
        Mode::RenameSession { .. } => [BTN_RENAME, BTN_CANCEL],
        Mode::ConfirmDelete { .. } => [BTN_DELETE, BTN_CANCEL],
        Mode::FilePreview { .. } => [BTN_CLOSE, ""],
        Mode::Normal => return None,
    };

    if is_preview {
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);
        let actions = action_rects(rows[2], &labels, None);
        Some(ModalLayout {
            area,
            instruction: rows[0],
            input: None,
            feedback: None,
            body: rows[1],
            actions,
            labels,
        })
    } else if matches!(mode, Mode::ConfirmDelete { .. }) {
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);
        let actions = action_rects(rows[2], &labels, Some(labels[1]));
        Some(ModalLayout {
            area,
            instruction: rows[0],
            input: None,
            feedback: None,
            body: rows[1],
            actions,
            labels,
        })
    } else {
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);
        let actions = action_rects(rows[4], &labels, Some(labels[1]));
        Some(ModalLayout {
            area,
            instruction: rows[0],
            input: Some(rows[1]),
            feedback: Some(rows[2]),
            body: rows[3],
            actions,
            labels,
        })
    }
}

fn action_rects(area: Rect, labels: &[&'static str; 2], second: Option<&str>) -> [Rect; 2] {
    let w1 = (labels[0].chars().count() + 2) as u16;
    let w2 = second.map(|value| (value.chars().count() + 2) as u16);
    match w2 {
        Some(w2) => {
            let x2 = area.x + area.width.saturating_sub(w2);
            let x1 = x2
                .saturating_sub(w1 + 2)
                .max(area.x);
            [
                Rect::new(x1, area.y, w1.min(area.width), 1),
                Rect::new(x2, area.y, w2.min(area.width), 1),
            ]
        }
        None => [
            Rect::new(
                area.x + area.width.saturating_sub(w1),
                area.y,
                w1.min(area.width),
                1,
            ),
            Rect::ZERO,
        ],
    }
}

fn centered(screen: Rect, width_pct: u16, height: u16) -> Rect {
    let width = ((screen.width as u32 * width_pct as u32 / 100) as u16)
        .max(30)
        .min(screen.width)
        .max(1);
    let height = height.max(1).min(screen.height).max(1);
    Rect::new(
        screen.x + screen.width.saturating_sub(width) / 2,
        screen.y + screen.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

pub fn draw(
    frame: &mut Frame,
    app: &mut App,
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    generation: Option<&Generation<'_>>,
) {
    app.screen = frame.area();
    if app.context_dirty {
        app.refresh_context(model, template);
    }
    let layout = layout(frame.area(), app.tab, app.show_debug);
    render_header(frame, layout.header, app, generation);

    let mut cursor = None;
    match (app.tab, layout.areas) {
        (Tab::Chat, Areas::Chat { sessions, chat, files }) => {
            if let Some(rect) = sessions {
                render_sessions(frame, app, rect, generation);
            }
            let (history, input) = chat_parts(chat);
            cursor = render_chat(frame, app, history, input, generation);
            if let Some(rect) = files {
                render_files(frame, app, rect);
            }
        }
        (Tab::Context, Areas::Context { stats, text }) => {
            render_context(frame, app, stats, text, generation);
        }
        (Tab::Debug, Areas::Debug { toolbar, log }) => {
            render_debug(frame, app, toolbar, log);
        }
        (Tab::Help, Areas::Help { text }) => {
            render_help(frame, app, text);
        }
        _ => {}
    }

    if let Some(rect) = layout.debug {
        render_debug_strip(frame, app, rect);
    }
    render_footer(frame, layout.footer, app, generation);

    if !app.mode.is_normal() {
        cursor = render_modal(frame, app);
    }
    if let Some((x, y)) = cursor {
        frame.set_cursor_position(Position::new(x, y));
    }
}

fn render_header(
    frame: &mut Frame,
    rect: Rect,
    app: &App,
    generation: Option<&Generation<'_>>,
) {
    if rect.width == 0 || rect.height == 0 {
        return;
    }
    let rects = tab_rects(rect);
    let mut spans = Vec::new();
    if brand_width(rect) > 0 {
        spans.push(Span::styled(
            BRAND,
            Style::default()
                .fg(Color::White)
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(" "));
    }
    for (index, tab) in Tab::ALL.iter().enumerate() {
        let hovered = app
            .mouse
            .map(|(x, y)| rects[index].1.contains(Position::new(x, y)))
            .unwrap_or(false);
        let style = if *tab == app.tab {
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else if hovered {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::Gray)
        };
        spans.push(Span::styled(tab.label(), style));
        if index + 1 < Tab::ALL.len() {
            spans.push(Span::styled("│", Style::default().fg(MUTED)));
        }
    }
    let tabs_width = spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum::<usize>() as u16;
    frame.render_widget(
        Paragraph::new(Line::from(spans)),
        Rect::new(rect.x, rect.y, tabs_width.min(rect.width), 1),
    );

    let usage = context_usage(app, generation);
    let mut info = Vec::new();
    if let Some(job) = generation {
        info.push(Span::styled(
            format!("{} ", spinner(job.elapsed())),
            Style::default().fg(WARN),
        ));
    }
    info.push(Span::styled(
        truncate_chars(&app.model.name, 24),
        Style::default().fg(ACCENT),
    ));
    info.push(Span::styled(" · ", Style::default().fg(MUTED)));
    info.push(Span::styled(
        truncate_chars(&app.current().name, 24),
        Style::default().fg(WARN),
    ));
    info.push(Span::styled(" · ", Style::default().fg(MUTED)));
    info.push(Span::styled("ctx ", Style::default().fg(MUTED)));
    let filled = (usage.percent().min(100) * 8).div_ceil(100);
    info.push(Span::styled(
        "▰".repeat(filled),
        Style::default().fg(usage.color()),
    ));
    info.push(Span::styled(
        "▱".repeat(8 - filled),
        Style::default().fg(MUTED),
    ));
    info.push(Span::styled(
        format!(" {}%", usage.percent()),
        Style::default()
            .fg(usage.color())
            .add_modifier(Modifier::BOLD),
    ));
    let info_width = info
        .iter()
        .map(|span| span.content.chars().count())
        .sum::<usize>() as u16;
    let start = rect.x + tabs_width;
    if start + info_width + 1 < rect.right() {
        frame.render_widget(
            Paragraph::new(Line::from(info)),
            Rect::new(rect.right() - info_width, rect.y, info_width, 1),
        );
    }
}

fn render_footer(
    frame: &mut Frame,
    rect: Rect,
    app: &App,
    generation: Option<&Generation<'_>>,
) {
    if rect.width == 0 || rect.height == 0 {
        return;
    }
    let context_tokens = generation
        .filter(|job| job.session == app.active)
        .map(|job| job.prompt_tokens)
        .unwrap_or(app.context.total_tokens);
    let location = format!(
        "{} · {} · {} msgs · {}/{} ctx",
        app.tab.title(),
        truncate_chars(&app.current().name, 22),
        app.current().messages.len(),
        context_tokens,
        app.model.n_ctx
    );
    let hints = footer_hints(app, generation);
    if rect.height >= 2 {
        render_dual(
            frame,
            Rect::new(rect.x, rect.y, rect.width, 1),
            hints,
            Style::default().fg(MUTED),
            location,
            Style::default().fg(MUTED),
        );
        let (icon, color) = match app.status_kind {
            StatusKind::Info => ("●", ACCENT),
            StatusKind::Ok => ("●", OK),
            StatusKind::Warn => ("●", WARN),
            StatusKind::Error => ("●", ERROR),
        };
        let left = format!("{icon} {}", app.status);
        render_dual(
            frame,
            Rect::new(rect.x, rect.y + 1, rect.width, 1),
            left,
            Style::default().fg(color),
            generation_status(app, generation),
            Style::default().fg(if generation.is_some() { WARN } else { MUTED }),
        );
    } else {
        let left = format!("● {}", app.status);
        render_dual(
            frame,
            rect,
            left,
            Style::default().fg(ACCENT),
            location,
            Style::default().fg(MUTED),
        );
    }
}

fn footer_hints(app: &App, generation: Option<&Generation<'_>>) -> String {
    if !app.mode.is_normal() {
        return "Enter confirma · Esc cancela · ratón: botones del diálogo".into();
    }
    let mut hints: String = match app.tab {
        Tab::Chat => {
            if app.input.is_empty() {
                "Enter enviar · Ctrl-F archivo · Ctrl-N sesión · Ctrl-R renombrar · Ctrl-D borrar · Ctrl-S guardar · Ctrl-G debug · Tab/F1-F4 pestañas · Ctrl-Q salir".into()
            } else {
                "Enter enviar · Ctrl-A/E inicio/fin · Ctrl-W borra palabra · Tab/F1-F4 pestañas · Ctrl-Q salir".into()
            }
        }
        Tab::Context => {
            "F5 recarga archivos y recalcula · rueda/PgUp/PgDn desplazan · F1 chat · clic en pestañas".into()
        }
        Tab::Debug => {
            "clic [Limpiar]/[Seguir] · rueda/PgUp/PgDn desplazan · F1 chat · clic en pestañas".into()
        }
        Tab::Help => "rueda/PgUp/PgDn desplazan · F1-F4 o clic en pestañas · Ctrl-Q salir".into(),
    };
    if generation.is_some() {
        hints.push_str(" · Esc cancela generación");
    }
    hints
}

fn generation_status(app: &App, generation: Option<&Generation<'_>>) -> String {
    if let Some(job) = generation {
        if job.prefill_done {
            format!(
                "{} · gen {}/{} · {:.1} t/s",
                app.sessions
                    .get(job.session)
                    .map(|session| session.name.as_str())
                    .unwrap_or("?"),
                job.produced,
                job.max_new,
                job.tokens_per_second()
            )
        } else {
            format!(
                "{} · prefill {}/{} · {:.1}s",
                app.sessions
                    .get(job.session)
                    .map(|session| session.name.as_str())
                    .unwrap_or("?"),
                job.prefill_pos,
                job.prefill_total,
                job.elapsed().as_secs_f64()
            )
        }
    } else if let Some(stats) = &app.last_stats {
        format!(
            "último: {} tok · {:.1} t/s · {} ms",
            stats.generated, stats.tok_per_s, stats.total_ms
        )
    } else {
        format!(
            "ctx {}/{} · máx {} nuevos",
            app.context.total_tokens, app.model.n_ctx, app.model.max_tokens
        )
    }
}

fn render_dual(
    frame: &mut Frame,
    rect: Rect,
    left: String,
    left_style: Style,
    right: String,
    right_style: Style,
) {
    if rect.width == 0 || rect.height == 0 {
        return;
    }
    let right_width = (display_width(&right) as u16).min(rect.width);
    let gap = if right_width > 0 && right_width < rect.width {
        1
    } else {
        0
    };
    let left_width = rect.width.saturating_sub(right_width + gap);
    if left_width > 0 {
        frame.render_widget(
            Paragraph::new(left).style(left_style),
            Rect::new(rect.x, rect.y, left_width, rect.height),
        );
    }
    if right_width > 0 {
        frame.render_widget(
            Paragraph::new(right)
                .style(right_style)
                .alignment(Alignment::Right),
            Rect::new(rect.x + rect.width - right_width, rect.y, right_width, 1),
        );
    }
}

fn render_sessions(
    frame: &mut Frame,
    app: &mut App,
    rect: Rect,
    generation: Option<&Generation<'_>>,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, rect))
        .title(format!(
            " Sesiones · {}/{} ",
            app.active + 1,
            app.sessions.len()
        ));
    frame.render_widget(block, rect);
    let parts = sidebar_rects(rect);

    if parts.new.height > 0 && parts.new.width > 0 {
        let label = truncate_chars("+ Nueva sesión", parts.new.width as usize);
        let style = if is_hovered(app, parts.new) {
            Style::default()
                .fg(Color::Black)
                .bg(OK)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(OK).add_modifier(Modifier::BOLD)
        };
        frame.render_widget(Paragraph::new(pad(&label, parts.new.width)).style(style), parts.new);
    }

    let visible = parts.list.height as usize;
    if visible > 0 {
        if app.active < app.session_offset as usize {
            app.session_offset = app.active as u16;
        }
        if app.active >= app.session_offset as usize + visible {
            app.session_offset = (app.active + 1 - visible) as u16;
        }
        let max_offset = app.sessions.len().saturating_sub(visible);
        app.session_offset = app.session_offset.min(max_offset as u16);
        let start = app.session_offset as usize;
        let mut lines = Vec::new();
        for index in start..(start + visible).min(app.sessions.len()) {
            let session = &app.sessions[index];
            let active = index == app.active;
            let marker = if active { "▸" } else { " " };
            let spinner_text = generation
                .filter(|job| job.session == index)
                .map(|job| format!("{} ", spinner(job.elapsed())))
                .unwrap_or_default();
            let count = format!("{}m", session.messages.len());
            let prefix_width = 2 + display_width(&spinner_text) + count.chars().count() + 1;
            let name_width = parts
                .list
                .width
                .saturating_sub(prefix_width as u16)
                .max(4);
            let name = truncate_chars(&session.name, name_width as usize);
            let padding = name_width.saturating_sub(display_width(&name) as u16);
            let mut spans = vec![Span::styled(
                format!("{marker} "),
                Style::default().fg(if active { WARN } else { MUTED }),
            )];
            if !spinner_text.is_empty() {
                spans.push(Span::styled(spinner_text, Style::default().fg(WARN)));
            }
            spans.push(Span::styled(
                name,
                if active {
                    Style::default().fg(WARN).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                },
            ));
            spans.push(Span::raw(" ".repeat(padding as usize)));
            spans.push(Span::styled(count, Style::default().fg(MUTED)));
            lines.push(Line::from(spans));
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), parts.list);
    }

    if parts.rename.height > 0 {
        let rename_style = if is_hovered(app, parts.rename) {
            Style::default().fg(Color::Black).bg(ACCENT)
        } else {
            Style::default().fg(Color::Gray)
        };
        let delete_style = if is_hovered(app, parts.delete) {
            Style::default().fg(Color::Black).bg(ERROR)
        } else {
            Style::default().fg(MUTED)
        };
        render_filled(
            frame,
            parts.rename,
            "Renombrar",
            rename_style,
        );
        render_filled(frame, parts.delete, "Borrar", delete_style);
    }
}

fn render_files(frame: &mut Frame, app: &mut App, rect: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, rect))
        .title(format!(" Archivos · {} ", app.current().selected_files.len()));
    frame.render_widget(block, rect);
    let parts = files_rects(rect);

    if parts.add.height > 0 && parts.add.width > 0 {
        let text = if parts.add.width >= 24 {
            "+ Añadir archivo (Ctrl-F)"
        } else {
            "+ Añadir"
        };
        let label = truncate_chars(text, parts.add.width as usize);
        let style = if is_hovered(app, parts.add) {
            Style::default()
                .fg(Color::Black)
                .bg(OK)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(OK).add_modifier(Modifier::BOLD)
        };
        frame.render_widget(Paragraph::new(pad(&label, parts.add.width)).style(style), parts.add);
    }

    let visible = parts.list.height as usize;
    if visible > 0 {
        let files = app.current().selected_files.clone();
        let max_offset = files.len().saturating_sub(visible);
        app.file_offset = app.file_offset.min(max_offset as u16);
        let start = app.file_offset as usize;
        let mut lines = Vec::new();
        if files.is_empty() {
            lines.push(Line::from(Span::styled(
                "Sin archivos",
                Style::default().fg(MUTED),
            )));
            if visible > 1 {
                lines.push(Line::from(Span::styled(
                    "Ctrl-F o + Añadir",
                    Style::default().fg(MUTED),
                )));
            }
        } else {
            for index in start..(start + visible).min(files.len()) {
                let path = &files[index];
                let info = app.file_cache.get(path);
                let tokens = info.map(|value| value.tokens).unwrap_or(0);
                let failed = info.map(|value| value.error).unwrap_or(false);
                let suffix = format!("{tokens}t");
                let name_width = parts
                    .list
                    .width
                    .saturating_sub(4 + suffix.chars().count() as u16 + 1)
                    .max(4);
                let name = truncate_chars(path, name_width as usize);
                let padding = name_width.saturating_sub(display_width(&name) as u16);
                lines.push(Line::from(vec![
                    Span::styled("[x]", Style::default().fg(ERROR).add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(
                        name,
                        Style::default().fg(if failed { ERROR } else { Color::White }),
                    ),
                    Span::raw(" ".repeat(padding as usize)),
                    Span::styled(suffix, Style::default().fg(MUTED)),
                ]));
            }
        }
        frame.render_widget(Paragraph::new(Text::from(lines)), parts.list);
    }

    if parts.footer.height > 0 && parts.footer.width > 0 {
        let hint = if app.current().selected_files.is_empty() {
            "Ctrl-F añade archivos"
        } else {
            "clic [x] quita · clic ruta abre"
        };
        frame.render_widget(
            Paragraph::new(truncate_chars(hint, parts.footer.width as usize))
                .style(Style::default().fg(MUTED)),
            parts.footer,
        );
    }
}

fn render_chat(
    frame: &mut Frame,
    app: &mut App,
    history: Rect,
    input: Rect,
    generation: Option<&Generation<'_>>,
) -> Option<(u16, u16)> {
    let active_generation = generation.filter(|job| job.session == app.active);
    let mut lines = Vec::new();
    let messages = &app.current().messages;
    if messages.is_empty() && active_generation.is_none() {
        lines.push(Line::from(Span::styled(
            "Bienvenido a LLM Cheap",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for text in [
            "Escribe abajo y pulsa Enter. Adjunta archivos con Ctrl-F o el botón + Añadir.",
            "F2 Contexto muestra exactamente lo que recibe el modelo.",
            "F3 Debug contiene el registro completo; Ctrl-G muestra/oculta un panel bajo el chat.",
            "La línea inferior del chat mide el contexto: sistema, historial, archivos, mensaje y respuesta.",
            "Ratón: clic en pestañas, sesiones, archivos y botones; rueda para desplazar.",
        ] {
            lines.push(Line::from(Span::styled(text, Style::default().fg(MUTED))));
        }
        lines.push(Line::from(""));
    }
    for message in messages {
        push_message(&mut lines, &message.role, &message.content, false);
    }
    if let Some(job) = active_generation {
        if !job.prompt.is_empty() {
            push_message(&mut lines, "user", &job.prompt, false);
        }
        if !job.prefill_done {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} Asistente", spinner(job.elapsed())),
                    Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        "  · preparando contexto {}/{}",
                        job.prefill_pos, job.prefill_total
                    ),
                    Style::default().fg(MUTED),
                ),
            ]));
            lines.push(Line::from(Span::styled(
                "  El prompt se está evaluando por bloques; la interfaz sigue activa.",
                Style::default().fg(MUTED),
            )));
        } else {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} Asistente", spinner(job.elapsed())),
                    Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(
                        "  · generando {}/{} · {:.1} t/s",
                        job.produced,
                        job.max_new,
                        job.tokens_per_second()
                    ),
                    Style::default().fg(MUTED),
                ),
            ]));
            if job.produced == 0 {
                lines.push(Line::from(Span::styled(
                    "  Pensando…",
                    Style::default().fg(MUTED),
                )));
            } else {
                for text in job.response_text().lines() {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(text.to_owned(), Style::default().fg(Color::White)),
                    ]));
                }
                lines.push(Line::from(Span::styled("  ▌", Style::default().fg(WARN))));
            }
        }
        lines.push(Line::from(""));
    }

    let title = if active_generation.is_some() {
        format!(" Chat · {} · generando ", app.current().name)
    } else {
        format!(" Chat · {} ", app.current().name)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, history))
        .title(title)
        .title_bottom(context_meter(
            &context_usage(app, generation),
            history.width.saturating_sub(4) as usize,
        ));
    let paragraph = Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .block(block);
    let total = paragraph.line_count(history.width.saturating_sub(2));
    let max_scroll = total.saturating_sub(history.height as usize);
    let offset = (app.chat_offset as usize).min(max_scroll);
    app.chat_offset = offset as u16;
    let scroll = max_scroll.saturating_sub(offset);
    frame.render_widget(paragraph.scroll((scroll.min(u16::MAX as usize) as u16, 0)), history);

    let generating = active_generation.is_some();
    let (inner, cursor) = render_input(frame, app, input, generating);
    cursor.map(|cursor_x| (inner.x + cursor_x, inner.y))
}

fn render_input(
    frame: &mut Frame,
    app: &App,
    rect: Rect,
    generating: bool,
) -> (Rect, Option<u16>) {
    if rect.width == 0 || rect.height == 0 {
        return (rect, None);
    }
    let inner = if rect.height >= 3 {
        let title = if generating {
            " Mensaje · generación en curso · Esc cancela "
        } else {
            " Mensaje · Enter envía · Ctrl-F archivo "
        };
        let style = if generating {
            Style::default().fg(WARN)
        } else if is_hovered(app, rect) {
            Style::default().fg(ACCENT)
        } else {
            Style::default().fg(MUTED)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(style)
            .title(title);
        let inner = block.inner(rect);
        frame.render_widget(block, rect);
        inner
    } else {
        rect
    };
    if inner.width == 0 || inner.height == 0 {
        return (inner, None);
    }
    let (shown, cursor_x) = visible_input(&app.input, app.cursor, inner.width);
    let (text, style) = if app.input.is_empty() {
        (
            "Escribe un mensaje…".to_owned(),
            Style::default().fg(MUTED),
        )
    } else {
        (shown, Style::default().fg(Color::White))
    };
    frame.render_widget(Paragraph::new(text).style(style), inner);
    (inner, Some(cursor_x))
}

fn push_message(lines: &mut Vec<Line<'static>>, role: &str, content: &str, dim: bool) {
    let (label, color) = match role {
        "user" if content.starts_with("<tool_response>") => ("Herramienta", MUTED),
        "user" => ("Tú", USER),
        "assistant" => ("Asistente", ASSISTANT),
        _ => (role, MUTED),
    };
    let role_style = if dim {
        Style::default().fg(MUTED)
    } else {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    };
    lines.push(Line::from(Span::styled(format!("  {label}"), role_style)));
    if content.is_empty() {
        lines.push(Line::from(""));
    } else {
        for value in content.lines() {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    value.to_owned(),
                    Style::default().fg(if dim { MUTED } else { Color::White }),
                ),
            ]));
        }
    }
    lines.push(Line::from(""));
}

fn render_context(
    frame: &mut Frame,
    app: &mut App,
    stats_rect: Rect,
    text_rect: Rect,
    generation: Option<&Generation<'_>>,
) {
    let active_generation = generation.filter(|job| job.session == app.active);
    render_context_stats(frame, app, stats_rect, active_generation);

    let (text, title, tokens_style) = if let Some(job) = active_generation {
        (
            job.formatted.clone(),
            format!(" Contexto en ejecución · {} tokens ", job.prompt_tokens),
            Style::default().fg(WARN),
        )
    } else {
        (
            app.context.text.clone(),
            format!(
                " Contexto exacto (vista previa) · {} tokens ",
                app.context.total_tokens
            ),
            Style::default().fg(ACCENT),
        )
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, text_rect))
        .title(title)
        .title_style(tokens_style);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .block(block);
    let total = paragraph.line_count(text_rect.width.saturating_sub(2));
    let max_scroll = total.saturating_sub(text_rect.height as usize);
    let offset = (app.context_scroll as usize).min(max_scroll);
    app.context_scroll = offset as u16;
    frame.render_widget(
        paragraph.scroll((offset.min(u16::MAX as usize) as u16, 0)),
        text_rect,
    );
}

fn render_context_stats(
    frame: &mut Frame,
    app: &mut App,
    rect: Rect,
    generation: Option<&Generation<'_>>,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, rect))
        .title(" Resumen del contexto ");
    frame.render_widget(block, rect);
    let inner = rect.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let prompt_tokens = generation
        .map(|job| job.prompt_tokens)
        .unwrap_or(app.context.total_tokens);
    let ratio = (prompt_tokens as f64 / app.model.n_ctx.max(1) as f64).clamp(0.0, 1.0);
    let color = if ratio >= 0.9 {
        ERROR
    } else if ratio >= 0.7 {
        WARN
    } else {
        OK
    };
    let mut body_y = inner.y;
    let mut body_height = inner.height;
    if inner.height >= 3 {
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).bg(Color::Black))
            .ratio(ratio)
            .label(format!(
                "{prompt_tokens} / {} tokens ({:.0}%)",
                app.model.n_ctx,
                ratio * 100.0
            ));
        frame.render_widget(gauge, Rect::new(inner.x, inner.y, inner.width, 1));
        body_y += 2;
        body_height = inner.height.saturating_sub(2);
    }

    let mut lines = Vec::new();
    kv(&mut lines, "Sesión", &app.current().name, WARN);
    kv(
        &mut lines,
        "Mensajes",
        &format!(
            "{} · archivos {}",
            app.current().messages.len(),
            app.current().selected_files.len()
        ),
        Color::White,
    );
    lines.push(Line::from(""));
    if let Some(job) = generation {
        kv(
            &mut lines,
            "Prompt",
            &format!("{} tokens · n_ctx {}", job.prompt_tokens, app.model.n_ctx),
            Color::White,
        );
        kv(
            &mut lines,
            "Prefill",
            &if job.prefill_done {
                format!("completo en {} ms", job.prefill_ms)
            } else {
                format!("{}/{} tokens", job.prefill_pos, job.prefill_total)
            },
            Color::White,
        );
        kv(
            &mut lines,
            "Generando",
            &format!("{}/{} tokens", job.produced, job.max_new),
            if job.prefill_done { WARN } else { MUTED },
        );
        kv(
            &mut lines,
            "Velocidad",
            &format!("{:.1} t/s", job.tokens_per_second()),
            OK,
        );
        kv(&mut lines, "Modelo", &app.model.name, ACCENT);
        kv(
            &mut lines,
            "Ajustes",
            &format!(
                "temp {:.2} · top_p {:.2} · máx {} nuevos",
                app.model.temperature, app.model.top_p, job.max_new
            ),
            MUTED,
        );
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "El panel derecho es el prompt exacto que está evaluando el modelo.",
            Style::default().fg(MUTED),
        )));
    } else {
        kv(
            &mut lines,
            "Composición",
            &format!(
                "sistema {} · historial {} · archivos {} · entrada {}",
                app.context.system_tokens,
                app.context.history_tokens,
                app.context.files_tokens,
                app.context.input_tokens
            ),
            Color::White,
        );
        lines.push(Line::from(Span::styled(
            "Las secciones son aproximadas; el total del medidor es exacto.",
            Style::default().fg(MUTED),
        )));
        kv(
            &mut lines,
            "Libre",
            &format!(
                "{} tokens",
                app.model.n_ctx.saturating_sub(app.context.total_tokens)
            ),
            OK,
        );
        kv(
            &mut lines,
            "Máximo nuevo",
            &format!("{} tokens", app.model.max_tokens),
            Color::White,
        );
        kv(
            &mut lines,
            "n_ctx",
            &format!(
                "{} · entrenado {}",
                app.model.n_ctx, app.model.n_ctx_train
            ),
            MUTED,
        );
        kv(&mut lines, "Modelo", &app.model.name, ACCENT);
        let last = app
            .last_stats
            .as_ref()
            .map(|stats| {
                format!(
                    "{} tok · {:.1} t/s · {} ms",
                    stats.generated, stats.tok_per_s, stats.total_ms
                )
            })
            .unwrap_or_else(|| "—".into());
        kv(&mut lines, "Último turno", &last, if app.last_stats.is_some() { OK } else { MUTED });
        if let Some(note) = app.context.note.clone() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                note,
                Style::default().fg(ERROR).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "F5 recarga archivos · rueda desplaza el texto · Tab cambia pestaña",
            Style::default().fg(MUTED),
        )));
    }
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::White)),
        Rect::new(inner.x, body_y, inner.width, body_height),
    );
}

fn kv(lines: &mut Vec<Line<'static>>, key: &str, value: &str, color: Color) {
    lines.push(Line::from(vec![
        Span::styled(format!("{key}: "), Style::default().fg(MUTED)),
        Span::styled(value.to_owned(), Style::default().fg(color)),
    ]));
}

fn render_debug(frame: &mut Frame, app: &mut App, toolbar: Rect, log: Rect) {
    if toolbar.width > 0 {
        let rects = debug_toolbar_rects(toolbar);
        render_button(
            frame,
            rects.clear,
            BTN_CLEAR,
            Style::default()
                .fg(Color::Black)
                .bg(WARN)
                .add_modifier(Modifier::BOLD),
            is_hovered(app, rects.clear),
        );
        let follow_style = if app.debug_offset == 0 {
            Style::default()
                .fg(Color::Black)
                .bg(OK)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White).bg(MUTED)
        };
        render_button(
            frame,
            rects.follow,
            BTN_FOLLOW,
            follow_style,
            is_hovered(app, rects.follow),
        );
        let used = rects.follow.x + rects.follow.width;
        if used < toolbar.right() {
            let info = format!(
                "{} entradas · {}",
                app.logs.len(),
                if app.debug_offset == 0 {
                    "siguiendo"
                } else {
                    "pausado"
                }
            );
            frame.render_widget(
                Paragraph::new(info)
                    .alignment(Alignment::Right)
                    .style(Style::default().fg(MUTED)),
                Rect::new(used, toolbar.y, toolbar.right() - used, 1),
            );
        }
    }
    render_log_panel(frame, app, log, true);
}

fn render_debug_strip(frame: &mut Frame, app: &mut App, rect: Rect) {
    render_log_panel(frame, app, rect, false);
}

fn render_log_panel(frame: &mut Frame, app: &mut App, rect: Rect, full: bool) {
    let title = if full {
        format!(" Registro de depuración · {} entradas ", app.logs.len())
    } else {
        " Depuración · clic abre F3 · Ctrl-G oculta ".to_owned()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, rect))
        .title(title);
    let lines = log_lines(app);
    let paragraph = Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .block(block);
    let total = paragraph.line_count(rect.width.saturating_sub(2));
    let max_scroll = total.saturating_sub(rect.height as usize);
    let offset = (app.debug_offset as usize).min(max_scroll);
    app.debug_offset = offset as u16;
    let scroll = max_scroll.saturating_sub(offset);
    frame.render_widget(
        paragraph.scroll((scroll.min(u16::MAX as usize) as u16, 0)),
        rect,
    );
}

fn log_lines(app: &App) -> Vec<Line<'static>> {
    if app.logs.is_empty() {
        return vec![Line::from(Span::styled(
            "Sin eventos todavía.",
            Style::default().fg(MUTED),
        ))];
    }
    let mut lines = Vec::new();
    for entry in &app.logs {
        let color = match entry.level {
            LogLevel::Info => ACCENT,
            LogLevel::Ok => OK,
            LogLevel::Warn => WARN,
            LogLevel::Error => ERROR,
            LogLevel::Gen => Color::Magenta,
        };
        let mut parts = entry.message.lines();
        let first = parts.next().unwrap_or("");
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", entry.at), Style::default().fg(MUTED)),
            Span::styled(
                format!("{:<4} ", entry.level.label()),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(first.to_owned(), Style::default().fg(Color::White)),
        ]));
        for rest in parts {
            lines.push(Line::from(vec![
                Span::raw("        "),
                Span::styled(rest.to_owned(), Style::default().fg(Color::Gray)),
            ]));
        }
    }
    lines
}

fn render_help(frame: &mut Frame, app: &mut App, rect: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(pane_style(app, rect))
        .title(" Ayuda · teclado, ratón y comandos ");
    let mut lines = Vec::new();
    section(&mut lines, "Dónde estás");
    bullet(&mut lines, "La pestaña activa está resaltada en la cabecera y el pie muestra la sesión, mensajes y % de contexto.");
    bullet(&mut lines, "Tab / Shift-Tab o F1-F4 cambian de pestaña. También puedes hacer clic en las pestañas.");
    bullet(&mut lines, "Chat = conversación · Contexto = prompt exacto · Debug = registro · Ayuda = esta página.");
    lines.push(Line::from(""));

    section(&mut lines, "Sesiones");
    bullet(&mut lines, "Ctrl-N crea una sesión y Ctrl-P / Ctrl-L cambian a la anterior / siguiente.");
    bullet(&mut lines, "Clic en una sesión para activarla. Ctrl-R la renombra y Ctrl-D la elimina (con confirmación).");
    bullet(&mut lines, "Ctrl-S guarda. Las sesiones se guardan automáticamente al cambiar y en .llm-cheap/sessions/.");
    lines.push(Line::from(""));

    section(&mut lines, "Archivos");
    bullet(&mut lines, "Ctrl-F (o + Añadir) abre un buscador de archivos del proyecto: escribe para filtrar, ↑/↓ y Enter o clic para añadir.");
    bullet(&mut lines, "Clic en [x] quita el archivo. Clic en la ruta abre una vista previa con tamaño y tokens.");
    bullet(&mut lines, "Los archivos seleccionados se envían con tu mensaje.");
    lines.push(Line::from(""));

    section(&mut lines, "Herramientas");
    bullet(&mut lines, "El modelo puede listar carpetas, cambiar de directorio (cd), leer y escribir archivos dentro del proyecto.");
    bullet(&mut lines, "Si pide ejecutar un comando, Ctrl-Y lo aprueba y Esc lo rechaza. Máximo 8 pasos por mensaje.");
    lines.push(Line::from(""));

    section(&mut lines, "Chat y generación");
    bullet(&mut lines, "Enter envía. Esc cancela una generación en curso (el mensaje vuelve al campo).");
    bullet(&mut lines, "Arriba/Abajo y PgUp/PgDn desplazan el chat; la rueda del ratón también funciona.");
    bullet(&mut lines, "Ctrl-A / Ctrl-E inicio/fin · Ctrl-U limpia · Ctrl-W borra palabra.");
    lines.push(Line::from(""));

    section(&mut lines, "Contexto y depuración");
    bullet(&mut lines, "F2 muestra el prompt exacto que recibe el modelo, el medidor de contexto y la composición.");
    bullet(&mut lines, "Durante una generación, F2 muestra el prompt que está evaluando el modelo en ese momento.");
    bullet(&mut lines, "F5 recarga los archivos y recalcula el contexto. F3 abre el registro completo con niveles y tiempos.");
    bullet(&mut lines, "Ctrl-G muestra u oculta el panel de depuración bajo el chat (oculto por defecto).");
    bullet(&mut lines, "Medidor bajo el chat: azul sistema · cian historial · magenta archivos · amarillo mensaje · verde respuesta · gris libre.");
    bullet(&mut lines, "El panel de depuración del chat muestra el final del registro; clic en él abre Debug.");
    lines.push(Line::from(""));

    section(&mut lines, "Comandos en el chat");
    for command in [
        "/archivo ruta · /quitar-archivo · /nueva [nombre] · /renombrar nombre",
        "/borrar · /guardar · /limpiar · /contexto · /debug · /ayuda · /chat · /salir",
    ] {
        bullet(&mut lines, command);
    }
    lines.push(Line::from(""));

    section(&mut lines, "Ratón");
    bullet(&mut lines, "Clic: pestañas, sesiones, botones, archivos, panel de debug y diálogos.");
    bullet(&mut lines, "Rueda: desplaza el panel bajo el puntero (chat, contexto, debug, ayuda o vista previa).");
    bullet(&mut lines, "El borde del panel bajo el puntero se ilumina para indicar dónde actuará la rueda.");
    lines.push(Line::from(""));

    section(&mut lines, "Modelo");
    bullet(&mut lines, &format!("Archivo: {}", app.model.path));
    bullet(&mut lines, &format!(
        "n_ctx {} (entrenado {}) · máximo nuevo {} · temp {:.2} · top_p {:.2}",
        app.model.n_ctx, app.model.n_ctx_train, app.model.max_tokens, app.model.temperature, app.model.top_p
    ));
    bullet(&mut lines, "Variables: MODEL_PATH, N_CTX, N_GPU_LAYERS, MAX_TOKENS, TEMPERATURE, TOP_P.");

    let paragraph = Paragraph::new(Text::from(lines))
        .wrap(Wrap { trim: false })
        .block(block);
    let total = paragraph.line_count(rect.width.saturating_sub(2));
    let max_scroll = total.saturating_sub(rect.height as usize);
    let offset = (app.help_scroll as usize).min(max_scroll);
    app.help_scroll = offset as u16;
    frame.render_widget(
        paragraph.scroll((offset.min(u16::MAX as usize) as u16, 0)),
        rect,
    );
}

fn section(lines: &mut Vec<Line<'static>>, title: &str) {
    lines.push(Line::from(Span::styled(
        title.to_owned(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
}

fn bullet(lines: &mut Vec<Line<'static>>, text: &str) {
    lines.push(Line::from(vec![
        Span::styled("  • ", Style::default().fg(MUTED)),
        Span::styled(text.to_owned(), Style::default().fg(Color::White)),
    ]));
}

fn render_modal(frame: &mut Frame, app: &mut App) -> Option<(u16, u16)> {
    let mode = app.mode.clone();
    let layout = modal_layout(app.screen, &mode)?;
    frame.render_widget(Clear, layout.area);
    let (title, border) = match &mode {
        Mode::AddFile { .. } => (" Añadir archivo ".to_owned(), ACCENT),
        Mode::RenameSession { .. } => (" Renombrar sesión ".to_owned(), ACCENT),
        Mode::ConfirmDelete { .. } => (" Eliminar sesión ".to_owned(), ERROR),
        Mode::FilePreview { path, error, .. } => (
            format!(" {} ", truncate_chars(path, 70)),
            if *error { ERROR } else { ACCENT },
        ),
        Mode::Normal => return None,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(title);
    frame.render_widget(block, layout.area);

    let mut preview_scroll = None;
    let cursor = match &mode {
        Mode::AddFile {
            input,
            cursor,
            error,
            selected,
        } => {
            let matches = app.file_matches(input);
            frame.render_widget(
                Paragraph::new("Escribe para filtrar · ↑/↓ elige · Enter o clic añade · Esc cierra")
                    .style(Style::default().fg(MUTED)),
                layout.instruction,
            );
            let input_rect = layout.input.unwrap_or(layout.body);
            let (shown, cursor_x) = visible_input(input, *cursor, input_rect.width.saturating_sub(2));
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("› ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
                    Span::styled(shown, Style::default().fg(Color::White)),
                ])),
                input_rect,
            );
            if let Some(feedback) = layout.feedback {
                let (text, style) = match error {
                    Some(error) => (error.clone(), Style::default().fg(ERROR)),
                    None => (
                        format!("{} de {} archivos", matches.len(), app.project_files.len()),
                        Style::default().fg(MUTED),
                    ),
                };
                frame.render_widget(
                    Paragraph::new(truncate_chars(&text, feedback.width as usize)).style(style),
                    feedback,
                );
            }
            render_file_picker(frame, app, layout.body, &matches, *selected);
            render_modal_actions(frame, app, &layout, &mode);
            Some((input_rect.x + 2 + cursor_x, input_rect.y))
        }
        Mode::RenameSession { input, cursor, error } => {
            let instruction = "Nuevo nombre para esta sesión";
            frame.render_widget(
                Paragraph::new(instruction).style(Style::default().fg(MUTED)),
                layout.instruction,
            );
            let input_rect = layout.input.unwrap_or(layout.body);
            let (shown, cursor_x) = visible_input(input, *cursor, input_rect.width);
            frame.render_widget(
                Paragraph::new(shown).style(Style::default().fg(Color::White)),
                input_rect,
            );
            if let Some(feedback) = layout.feedback {
                let (text, style) = if let Some(error) = error {
                    (error.clone(), Style::default().fg(ERROR))
                } else {
                    (
                        "Enter confirma · Esc cancela".to_owned(),
                        Style::default().fg(MUTED),
                    )
                };
                frame.render_widget(
                    Paragraph::new(truncate_chars(&text, feedback.width as usize)).style(style),
                    feedback,
                );
            }
            render_modal_actions(frame, app, &layout, &mode);
            Some((input_rect.x + cursor_x, input_rect.y))
        }
        Mode::ConfirmDelete { index, selected } => {
            let name = app
                .sessions
                .get(*index)
                .map(|session| session.name.clone())
                .unwrap_or_else(|| "?".into());
            let count = app
                .sessions
                .get(*index)
                .map(|session| session.messages.len())
                .unwrap_or(0);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("¿Eliminar ", Style::default().fg(Color::White)),
                    Span::styled(name, Style::default().fg(WARN).add_modifier(Modifier::BOLD)),
                    Span::styled("?", Style::default().fg(Color::White)),
                ])),
                layout.instruction,
            );
            frame.render_widget(
                Paragraph::new(format!(
                    "Se borrarán {count} mensajes. Esta acción no se puede deshacer.\n\n←/→ elige · Enter confirma · Esc cancela"
                ))
                .style(Style::default().fg(MUTED))
                .wrap(Wrap { trim: false }),
                layout.body,
            );
            let delete_style = if *selected == 0 {
                Style::default()
                    .fg(Color::White)
                    .bg(ERROR)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray).bg(MUTED)
            };
            let cancel_style = if *selected == 1 {
                Style::default()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray).bg(MUTED)
            };
            render_button(
                frame,
                layout.actions[0],
                layout.labels[0],
                delete_style,
                is_hovered(app, layout.actions[0]),
            );
            render_button(
                frame,
                layout.actions[1],
                layout.labels[1],
                cancel_style,
                is_hovered(app, layout.actions[1]),
            );
            None
        }
        Mode::FilePreview {
            content,
            tokens,
            bytes,
            scroll,
            error,
            ..
        } => {
            frame.render_widget(
                Paragraph::new(format!(
                    "{} B · {} tokens · {}",
                    bytes,
                    tokens,
                    if *error {
                        "no se pudo leer"
                    } else {
                        "vista previa"
                    }
                ))
                .style(Style::default().fg(if *error { ERROR } else { MUTED })),
                layout.instruction,
            );
            let paragraph = Paragraph::new(content.clone())
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(if *error { ERROR } else { Color::White }));
            let total = paragraph.line_count(layout.body.width);
            let max_scroll = total.saturating_sub(layout.body.height as usize);
            let offset = (*scroll as usize).min(max_scroll);
            preview_scroll = Some(offset as u16);
            frame.render_widget(
                paragraph.scroll((offset.min(u16::MAX as usize) as u16, 0)),
                layout.body,
            );
            render_button(
                frame,
                layout.actions[0],
                layout.labels[0],
                Style::default()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD),
                is_hovered(app, layout.actions[0]),
            );
            None
        }
        Mode::Normal => None,
    };

    if let (Some(offset), Mode::FilePreview { scroll, .. }) = (preview_scroll, &mut app.mode) {
        *scroll = offset;
    }
    cursor
}

fn render_modal_actions(frame: &mut Frame, app: &App, layout: &ModalLayout, mode: &Mode) {
    let primary = match mode {
        Mode::AddFile { .. } => Style::default()
            .fg(Color::Black)
            .bg(OK)
            .add_modifier(Modifier::BOLD),
        Mode::RenameSession { .. } => Style::default()
            .fg(Color::Black)
            .bg(ACCENT)
            .add_modifier(Modifier::BOLD),
        _ => Style::default().fg(Color::White).bg(MUTED),
    };
    render_button(
        frame,
        layout.actions[0],
        layout.labels[0],
        primary,
        is_hovered(app, layout.actions[0]),
    );
    if layout.actions[1].width > 0 {
        render_button(
            frame,
            layout.actions[1],
            layout.labels[1],
            Style::default().fg(Color::White).bg(MUTED),
            is_hovered(app, layout.actions[1]),
        );
    }
}

fn render_button(
    frame: &mut Frame,
    rect: Rect,
    label: &str,
    style: Style,
    hovered: bool,
) {
    if rect.width == 0 || rect.height == 0 {
        return;
    }
    let style = if hovered {
        style.add_modifier(Modifier::REVERSED)
    } else {
        style
    };
    frame.render_widget(
        Paragraph::new(pad(label, rect.width))
            .style(style)
            .alignment(Alignment::Center),
        rect,
    );
}

fn render_filled(frame: &mut Frame, rect: Rect, label: &str, style: Style) {
    if rect.width == 0 || rect.height == 0 {
        return;
    }
    let width = rect.width as usize;
    let label = truncate_chars(label, width);
    let padding = width.saturating_sub(display_width(&label));
    let left = padding / 2;
    let text = format!("{}{}{}", " ".repeat(left), label, " ".repeat(padding - left));
    frame.render_widget(Paragraph::new(text).style(style), rect);
}

/// Primera fila visible del buscador para que `selected` quede en pantalla.
pub fn picker_start(selected: usize, visible: usize) -> usize {
    selected.saturating_sub(visible.saturating_sub(1))
}

fn render_file_picker(frame: &mut Frame, app: &App, rect: Rect, matches: &[String], selected: usize) {
    let visible = rect.height as usize;
    if visible == 0 || rect.width == 0 {
        return;
    }
    let mut lines = Vec::new();
    if matches.is_empty() {
        lines.push(Line::from(Span::styled(
            "Sin coincidencias · Enter intenta la ruta escrita",
            Style::default().fg(MUTED),
        )));
    }
    let start = picker_start(selected, visible);
    let chosen = &app.current().selected_files;
    for (offset, path) in matches.iter().skip(start).take(visible).enumerate() {
        let index = start + offset;
        let row = Rect::new(rect.x, rect.y + offset as u16, rect.width, 1);
        let added = chosen.contains(path);
        let tag = if added { " ✓ añadido" } else { "" };
        let name = truncate_chars(path, (rect.width as usize).saturating_sub(4 + tag.chars().count()));
        let style = if index == selected {
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else if is_hovered(app, row) {
            Style::default().fg(Color::White).add_modifier(Modifier::UNDERLINED)
        } else if added {
            Style::default().fg(MUTED)
        } else {
            Style::default().fg(Color::White)
        };
        let marker = if index == selected { "▸ " } else { "  " };
        lines.push(Line::from(vec![
            Span::styled(format!("{marker}{name}"), style),
            Span::styled(tag, Style::default().fg(OK)),
        ]));
    }
    frame.render_widget(Paragraph::new(Text::from(lines)), rect);
}

/// Tokens que ocupa cada parte del prompt, para el medidor de contexto.
struct ContextUsage {
    n_ctx: usize,
    /// (etiqueta, tokens, color, símbolo)
    segments: Vec<(&'static str, usize, Color, &'static str)>,
    used: usize,
}

impl ContextUsage {
    fn percent(&self) -> usize {
        (self.used * 100).checked_div(self.n_ctx.max(1)).unwrap_or(0)
    }

    fn color(&self) -> Color {
        match self.percent() {
            0..=69 => OK,
            70..=89 => WARN,
            _ => ERROR,
        }
    }
}

fn context_usage(app: &App, generation: Option<&Generation<'_>>) -> ContextUsage {
    let n_ctx = app.model.n_ctx;
    if let Some(job) = generation.filter(|job| job.session == app.active) {
        let pending = job.max_new.saturating_sub(job.produced);
        return ContextUsage {
            n_ctx,
            used: job.prompt_tokens + job.produced,
            segments: vec![
                ("prompt", job.prompt_tokens, SEG_HISTORY, "█"),
                ("respuesta", job.produced, SEG_REPLY, "█"),
                ("reserva", pending, SEG_REPLY, "▒"),
            ],
        };
    }
    let ctx = &app.context;
    let total = ctx.total_tokens;
    // El resto (plantilla de chat + instrucciones) cuenta como sistema.
    let system = total.saturating_sub(ctx.history_tokens + ctx.files_tokens + ctx.input_tokens);
    let reserve = app.model.max_tokens.min(n_ctx.saturating_sub(total));
    ContextUsage {
        n_ctx,
        used: total,
        segments: vec![
            ("sistema", system, SEG_SYSTEM, "█"),
            ("historial", ctx.history_tokens, SEG_HISTORY, "█"),
            ("archivos", ctx.files_tokens, SEG_FILES, "█"),
            ("mensaje", ctx.input_tokens, SEG_INPUT, "█"),
            ("respuesta", reserve, SEG_REPLY, "▒"),
        ],
    }
}

/// Línea `Contexto ████▒▒░░░ 1234/4096 30% · ■ historial 800 …` que cabe en `width`.
fn context_meter(usage: &ContextUsage, width: usize) -> Line<'static> {
    let n_ctx = usage.n_ctx.max(1);
    let summary = if usage.used >= n_ctx {
        format!(" {}/{} ¡no cabe! ", usage.used, n_ctx)
    } else {
        format!(" {}/{} {}% ", usage.used, n_ctx, usage.percent())
    };
    let fixed = " Contexto ".len() + summary.chars().count();
    // Leyenda con tokens; si no cabe, solo etiquetas; si tampoco, sin leyenda.
    let build = |numbers: bool| {
        usage
            .segments
            .iter()
            .filter(|(_, tokens, _, _)| *tokens > 0)
            .map(|(label, tokens, color, symbol)| {
                let text = if numbers {
                    format!(" {symbol}{label} {tokens}")
                } else {
                    format!(" {symbol}{label}")
                };
                (text, *color)
            })
            .collect::<Vec<_>>()
    };
    let width_of = |items: &[(String, Color)]| -> usize {
        items.iter().map(|(text, _)| text.chars().count()).sum()
    };
    let legend = [build(true), build(false)]
        .into_iter()
        .find(|items| width >= fixed + width_of(items) + 12)
        .unwrap_or_default();
    let legend_w = width_of(&legend);
    let with_legend = !legend.is_empty();
    let bar_w = width
        .saturating_sub(fixed + if with_legend { legend_w } else { 0 })
        .clamp(0, 40);

    let mut spans = vec![Span::styled(
        " Contexto ",
        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
    )];
    let mut cumulative = 0;
    let mut drawn = 0;
    for (_, tokens, color, symbol) in &usage.segments {
        cumulative += tokens;
        let end = (cumulative.min(n_ctx) * bar_w).div_ceil(n_ctx).min(bar_w);
        if end > drawn {
            spans.push(Span::styled(symbol.repeat(end - drawn), Style::default().fg(*color)));
            drawn = end;
        }
    }
    spans.push(Span::styled("░".repeat(bar_w - drawn), Style::default().fg(MUTED)));
    spans.push(Span::styled(
        summary,
        Style::default().fg(usage.color()).add_modifier(Modifier::BOLD),
    ));
    if with_legend {
        for (text, color) in legend {
            spans.push(Span::styled(text, Style::default().fg(color)));
        }
        spans.push(Span::raw(" "));
    }
    Line::from(spans)
}

fn pane_style(app: &App, rect: Rect) -> Style {
    if is_hovered(app, rect) {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(MUTED)
    }
}

fn is_hovered(app: &App, rect: Rect) -> bool {
    app.mouse
        .map(|(x, y)| rect.contains(Position::new(x, y)))
        .unwrap_or(false)
}

fn spinner(elapsed: Duration) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(elapsed.as_millis() / 100) as usize % FRAMES.len()]
}

fn visible_input(input: &str, cursor: usize, width: u16) -> (String, u16) {
    if width == 0 {
        return (String::new(), 0);
    }
    let chars: Vec<char> = input.chars().collect();
    let cursor = cursor.min(chars.len());
    let width = width as usize;
    let start = cursor.saturating_sub(width.saturating_sub(1));
    let shown = chars.iter().skip(start).take(width).collect::<String>();
    (
        shown,
        (cursor.saturating_sub(start)).min(width.saturating_sub(1)) as u16,
    )
}

fn pad(value: &str, width: u16) -> String {
    let width = width as usize;
    let value = truncate_chars(value, width);
    let padding = width.saturating_sub(display_width(&value));
    format!("{value}{}", " ".repeat(padding))
}

fn display_width(value: &str) -> usize {
    value
        .chars()
        .map(|ch| UnicodeWidthChar::width(ch).unwrap_or(1))
        .sum()
}
