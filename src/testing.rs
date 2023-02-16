use swc_common::sync::Lrc;
use swc_common::{FileName, SourceMap};
use swc_ecma_parser::lexer::Lexer;
use swc_ecma_parser::{Parser, StringInput, Syntax};

use super::parse::{Ast, parse};

pub(crate) fn parse_helper(contents: &str) -> Vec<Ast> {
    let cm = Lrc::<SourceMap>::default();
    let fm = cm.new_source_file(FileName::Custom("test.js".to_string()), contents.to_string());

    let lexer = Lexer::new(
        // We want to parse ecmascript
        Syntax::Es(Default::default()),
        // EsVersion defaults to es5
        Default::default(),
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);
    let body = parser.parse_module().expect("error parsing").body;
    parse(body)
}