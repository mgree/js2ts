use crate::parse::{Ast, AstNode, Type};

fn typecheck_helper(env: &mut Vec<(String, Type)>, ast: &Ast) -> Result<Type, String> {
    match &ast.ast {
        AstNode::Number(_) => Ok(Type::Number),
        AstNode::Boolean(_) => Ok(Type::Bool),
        AstNode::Binary { .. } => todo!(),
        AstNode::Unary { .. } => todo!(),

        AstNode::Ternary { cond, then, elsy } => {
            if typecheck_helper(env, &**cond)? != Type::Bool {
                return Err("condition of ternary operator must be of type `bool`".to_string());
            }

            let t1 = typecheck_helper(env, &**then)?;

            if t1 != typecheck_helper(env, &**elsy)? {
                Err("branches of ternary operator must be of the same type".to_string())
            } else {
                Ok(t1)
            }
        }

        AstNode::Coercion {
            expr,
            source_type,
            dest_type,
        } => {
            if typecheck_helper(env, &**expr)? != *source_type {
                Err("coercion source type must match expression type".to_string())
            } else {
                Ok(dest_type.clone())
            }
        }

        AstNode::Declare { vars } => {
            for (var, init) in vars {
                if let Some(init) = init {
                    let type_ = typecheck_helper(env, init)?;
                    env.push((var.clone(), type_));
                }
            }

            Ok(Type::Unit)
        }

        AstNode::Identifier(id) => {
            for (var, type_) in env.iter().rev() {
                if var == id {
                    return Ok(type_.clone());
                }
            }

            Err(format!("variable `{}` not found", id))
        }

        AstNode::Assign { var, expr } => {
            let t1 = typecheck_helper(env, &**expr)?;
            for (id, t2) in env.iter().rev() {
                if id == var {
                    if t1 == *t2 {
                        return Ok(t1);
                    } else {
                        return Err("variable reassigned to a different type".to_string());
                    }
                }
            }

            env.push((var.clone(), t1.clone()));
            Ok(t1)
        }
    }
}

/// Performs type checking on a migrated [`Ast`] to make sure the resulting Ast is valid.
pub fn typecheck(asts: &[Ast]) -> Result<(), String> {
    let mut env = Vec::new();
    for ast in asts {
        typecheck_helper(&mut env, ast)?;
    }

    Ok(())
}
