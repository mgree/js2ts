use swc_common::{SourceMap, FileName};
use swc_common::errors::{Handler, ColorConfig};
use swc_common::sync::Lrc;
use swc_ecma_parser::{StringInput, Syntax, Parser};
use swc_ecma_parser::lexer::Lexer;

use clap::Parser as ClapParser;

use js2ts::parse;

#[derive(ClapParser)]
struct Args {
    filename: String,
}

fn main() {
    let args = Args::parse();
    let file = match std::fs::read_to_string(&args.filename) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("could not open file `{}`: {}", args.filename, e);
            std::process::exit(1);
        }
    };

    let cm = Lrc::<SourceMap>::default();
    let handler = Handler::with_tty_emitter(ColorConfig::Always, true, false, Some(cm.clone()));

    let fm = cm.new_source_file(
        FileName::Custom(args.filename.into()),
        file.into()
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

    let module = match parser.parse_module() {
        Ok(v) => v,
        Err(e) => {
            e.into_diagnostic(&handler).emit();
            std::process::exit(1);
        }
    };

    println!("{:?}", parse::parse(module.body));
}
