use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use llama_cpp_2::model::LlamaChatTemplate;
use serde_json::{Map, Value};

use crate::app::truncate_chars;

/// Pasos de herramienta seguidos que el modelo puede encadenar por mensaje del usuario.
pub const MAX_STEPS: usize = 8;
/// Recorte de cada resultado para que un `ls` o un archivo grande no llene el contexto.
const MAX_OUTPUT: usize = 3000;

/// Esquemas en el formato con el que se entrenó Qwen2.5 (Hermes): un JSON por línea.
const TOOLS: &str = r#"{"type":"function","function":{"name":"list_dir","description":"Lista el contenido de una carpeta","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Carpeta relativa al directorio actual; por defecto ."}}}}}
{"type":"function","function":{"name":"cd","description":"Cambia el directorio actual","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
{"type":"function","function":{"name":"read_file","description":"Lee un archivo de texto","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}
{"type":"function","function":{"name":"write_file","description":"Crea o reemplaza un archivo con el contenido completo","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}
{"type":"function","function":{"name":"run","description":"Ejecuta un comando de shell en el directorio actual (el usuario debe aprobarlo)","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}"#;

const INSTRUCTIONS: &str = "Eres un asistente local de programación con acceso a la carpeta del proyecto. Responde en español y sé breve.
Usa una herramienta solo cuando el usuario pida explorar, leer, editar archivos o ejecutar algo; si no, responde normalmente.
Tras cada <tool_response> continúa con otra herramienta o responde al usuario.";

/// Qwen3.5 (y Qwen3-Coder) llaman herramientas en XML; Qwen2.5 en JSON (Hermes).
/// Se detecta por la plantilla porque llama.cpp no aplica la jinja con herramientas.
pub fn uses_xml(template: &LlamaChatTemplate) -> bool {
    template.to_str().is_ok_and(|text| text.contains("<function="))
}

/// Mensaje de sistema con las herramientas, copiando lo que renderiza la plantilla del modelo.
pub fn system_prompt(cwd: &str, xml: bool) -> String {
    let intro = format!("{INSTRUCTIONS}\nDirectorio actual: {cwd}");
    if xml {
        format!(
            r#"# Tools

You have access to the following functions:

<tools>
{TOOLS}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

{intro}"#
        )
    } else {
        format!(
            r#"{intro}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{TOOLS}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"#
        )
    }
}

pub struct ToolCall {
    pub name: String,
    pub args: Value,
}

impl ToolCall {
    pub fn arg(&self, key: &str) -> Option<&str> {
        self.args.get(key).and_then(Value::as_str)
    }
}

/// `None` si la respuesta no pide herramienta; `Some(Err)` si la pide mal formada,
/// para devolverle el error al modelo y que lo corrija. Acepta XML (Qwen3.5) y JSON (Qwen2.5).
pub fn parse_tool_call(response: &str) -> Option<Result<ToolCall, String>> {
    let start = response.find("<tool_call>")? + "<tool_call>".len();
    let body = response[start..].split("</tool_call>").next().unwrap_or("").trim();
    if let Some(rest) = body.strip_prefix("<function=") {
        return Some(parse_xml(rest));
    }
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(err) => return Some(Err(format!("JSON inválido en <tool_call>: {err}"))),
    };
    let Some(name) = value.get("name").and_then(Value::as_str) else {
        return Some(Err("falta \"name\" en <tool_call>".into()));
    };
    // Algunos modelos mandan los argumentos como cadena JSON.
    let args = match value.get("arguments").cloned().unwrap_or(Value::Null) {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::Null),
        other => other,
    };
    Some(Ok(ToolCall {
        name: name.to_owned(),
        args,
    }))
}

/// `rest` empieza tras `<function=`: `name>\n<parameter=key>\nvalor\n</parameter>…`.
fn parse_xml(rest: &str) -> Result<ToolCall, String> {
    let (name, mut rest) = rest.split_once('>').ok_or("falta '>' tras <function=")?;
    let mut args = Map::new();
    while let Some(start) = rest.find("<parameter=") {
        let (key, after) = rest[start + "<parameter=".len()..]
            .split_once('>')
            .ok_or("falta '>' tras <parameter=")?;
        // Sin cierre = respuesta cortada por MAX_TOKENS; se usa lo que llegó.
        let (value, after) = after.split_once("</parameter>").unwrap_or((after, ""));
        // La plantilla pone exactamente un salto de línea alrededor del valor; el resto
        // (sangría de un archivo, por ejemplo) se conserva.
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        args.insert(key.trim().to_owned(), Value::String(value.to_owned()));
        rest = after;
    }
    Ok(ToolCall {
        name: name.trim().to_owned(),
        args: Value::Object(args),
    })
}

/// Resuelve `path` contra `cwd` (relativo a la raíz del proyecto) sin salir de ella.
// ponytail: comprobación léxica; un symlink dentro del proyecto podría apuntar fuera.
pub fn resolve(cwd: &str, path: &str) -> Result<PathBuf> {
    if Path::new(path).is_absolute() {
        bail!("usa rutas relativas al proyecto");
    }
    let mut out = PathBuf::new();
    for part in Path::new(cwd).join(path).components() {
        match part {
            Component::Normal(name) => out.push(name),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    bail!("la ruta sale del proyecto");
                }
            }
            _ => bail!("ruta no válida"),
        }
    }
    Ok(if out.as_os_str().is_empty() { PathBuf::from(".") } else { out })
}

/// Ejecuta todas las herramientas salvo `run`, que necesita aprobación del usuario.
pub fn execute(call: &ToolCall, cwd: &mut String) -> Result<String> {
    let target = resolve(cwd, call.arg("path").unwrap_or("."));
    match call.name.as_str() {
        "list_dir" => {
            let dir = target?;
            let mut names = fs::read_dir(&dir)
                .with_context(|| format!("no se pudo listar {}", dir.display()))?
                .filter_map(|entry| entry.ok())
                .map(|entry| {
                    let mut name = entry.file_name().to_string_lossy().into_owned();
                    if entry.path().is_dir() {
                        name.push('/');
                    }
                    name
                })
                .collect::<Vec<_>>();
            names.sort();
            Ok(if names.is_empty() { "(carpeta vacía)".into() } else { names.join("\n") })
        }
        "cd" => {
            let dir = target?;
            if !dir.is_dir() {
                bail!("no es una carpeta: {}", dir.display());
            }
            *cwd = dir.to_string_lossy().into_owned();
            Ok(format!("Directorio actual: {cwd}"))
        }
        "read_file" => {
            let file = target?;
            fs::read_to_string(&file).with_context(|| format!("no se pudo leer {}", file.display()))
        }
        "write_file" => {
            let file = target?;
            let content = call.arg("content").context("falta el argumento content")?;
            if let Some(parent) = file.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&file, content)
                .with_context(|| format!("no se pudo escribir {}", file.display()))?;
            Ok(format!("Escrito {} ({} bytes)", file.display(), content.len()))
        }
        other => bail!("herramienta desconocida: {other}"),
    }
}

// ponytail: bloquea la TUI hasta 30 s; pasar a un hilo si hacen falta comandos largos.
pub fn run(cwd: &str, command: &str) -> String {
    let output = Command::new("timeout")
        .args(["30", "sh", "-c", command])
        .current_dir(cwd)
        .output();
    match output {
        Ok(out) => format!(
            "código de salida: {}\n{}{}",
            out.status.code().map_or("señal".into(), |code| code.to_string()),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(err) => format!("error: no se pudo ejecutar: {err}"),
    }
}

/// Envuelve un resultado como lo espera la plantilla de Qwen (rol user + <tool_response>).
pub fn response(text: &str) -> String {
    format!("<tool_response>\n{}\n</tool_response>", truncate_chars(text.trim(), MAX_OUTPUT))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_stays_inside_project() {
        assert_eq!(resolve(".", "src/main.rs").unwrap(), PathBuf::from("src/main.rs"));
        assert_eq!(resolve("src", "..").unwrap(), PathBuf::from("."));
        assert_eq!(resolve("src", "../README.md").unwrap(), PathBuf::from("README.md"));
        assert!(resolve(".", "..").is_err());
        assert!(resolve("src", "../../etc").is_err());
        assert!(resolve(".", "/etc/passwd").is_err());
    }

    #[test]
    fn parses_tool_calls() {
        let call = parse_tool_call(
            "Voy a mirar.\n<tool_call>\n{\"name\": \"cd\", \"arguments\": {\"path\": \"src\"}}\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        assert_eq!(call.name, "cd");
        assert_eq!(call.arg("path"), Some("src"));
        let stringy = parse_tool_call(r#"<tool_call>{"name":"run","arguments":"{\"command\":\"ls\"}"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(stringy.arg("command"), Some("ls"));
        assert!(parse_tool_call("<tool_call>{roto</tool_call>").unwrap().is_err());
        assert!(parse_tool_call("hola").is_none());
        let xml = parse_tool_call(
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\ndef f():\n    return 1\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        assert_eq!(xml.name, "write_file");
        assert_eq!(xml.arg("path"), Some("a.py"));
        assert_eq!(xml.arg("content"), Some("def f():\n    return 1"));
    }
}
