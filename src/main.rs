use swc_common::{SourceMap, FileName};
use swc_common::errors::{Handler, ColorConfig};
use swc_common::sync::Lrc;
use swc_ecma_parser::{StringInput, Syntax, Parser};
use swc_ecma_parser::lexer::Lexer;

fn main() {
    let cm = Lrc::<SourceMap>::default();
    let handler = Handler::with_tty_emitter(ColorConfig::Always, true, true, Some(cm.clone()));

    let fm = cm.new_source_file(
        FileName::Custom("test.js".into()),
        "function foo() {}".into()
    );

   let lexer = Lexer::new(
        // We want to parse ecmascript
        Syntax::Es(Default::default()),
        // EsVersion defaults to es5
        Default::default(),
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    for e in parser.take_errors() {
        e.into_diagnostic(&handler).emit();
    }

    let _module = parser
        .parse_module()
        .map_err(|e| {
            // Unrecoverable fatal error occurred
            e.into_diagnostic(&handler).emit()
        })
        .expect("failed to parser module");
}
