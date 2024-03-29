use swc_ecma_ast::BinaryOp;

use crate::parse::{Ast, AstNode, Type};

fn typecheck_helper(env: &mut Vec<(String, Type)>, ast: &Ast, ret_type: Option<&Type>) -> Result<Type, String> {
    match &ast.ast {
        AstNode::Number(_) => Ok(Type::Number),
        AstNode::Boolean(_) => Ok(Type::Bool),

        AstNode::Binary { op, left, right } => {
            let left = typecheck_helper(env, &**left, ret_type)?;
            let right = typecheck_helper(env, &**right, ret_type)?;

            match op {
                BinaryOp::EqEqEq => todo!(),
                BinaryOp::NotEqEq => todo!(),

                BinaryOp::EqEq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq => {
                    if let (Type::Number, Type::Number) = (left, right) {
                        Ok(Type::Bool)
                    } else {
                        Err("binary comparison operation takes in numbers".to_string())
                    }
                }

                BinaryOp::LShift
                | BinaryOp::RShift
                | BinaryOp::ZeroFillRShift
                | BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::BitAnd => {
                    if let (Type::Number, Type::Number) = (left, right) {
                        Ok(Type::Number)
                    } else {
                        Err("binary number operation takes in numbers".to_string())
                    }
                }

                BinaryOp::LogicalOr => todo!(),
                BinaryOp::LogicalAnd => todo!(),
                BinaryOp::In => todo!(),
                BinaryOp::InstanceOf => todo!(),
                BinaryOp::Exp => todo!(),
                BinaryOp::NullishCoalescing => todo!(),
            }
        }

        AstNode::Unary { .. } => todo!(),

        AstNode::Ternary { cond, then, elsy } => {
            if typecheck_helper(env, &**cond, ret_type)? != Type::Bool {
                return Err("condition of ternary operator must be of type `bool`".to_string());
            }

            let t1 = typecheck_helper(env, &**then, ret_type)?;

            if t1 != typecheck_helper(env, &**elsy, ret_type)? {
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
            if typecheck_helper(env, &**expr, ret_type)? != *source_type {
                Err("coercion source type must match expression type".to_string())
            } else {
                Ok(dest_type.clone())
            }
        }

        AstNode::Declare { vars } => {
            for (var, init) in vars {
                if let Some(init) = init {
                    let type_ = typecheck_helper(env, init, ret_type)?;
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
            let t1 = typecheck_helper(env, &**expr, ret_type)?;
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

        AstNode::Block(block) => {
            let mut env = env.clone();
            for stat in block {
                typecheck_helper(&mut env, stat, ret_type)?;
            }
            Ok(Type::Unit)
        }

        AstNode::If { cond, then, elsy } => {
            let t = typecheck_helper(env, &**cond, ret_type)?;
            if t != Type::Bool {
                return Err("condition of if statement must be of type `bool`".to_string());
            }

            let mut env_ = env.clone();
            typecheck_helper(&mut env_, &**then, ret_type)?;
            if let Some(elsy) = elsy {
                let mut env_ = env.clone();
                typecheck_helper(&mut env_, &**elsy, ret_type)?;
            }

            Ok(Type::Unit)
        }

        AstNode::While { cond, body } => {
            let t = typecheck_helper(env, &**cond, ret_type)?;
            if t != Type::Bool {
                return Err("condition of if statement must be of type `bool`".to_string());
            }

            let mut env = env.clone();
            typecheck_helper(&mut env, &**body, ret_type)?;
            Ok(Type::Unit)
        }

        AstNode::FuncDecl {
            args,
            arg_types,
            ret_type,
            body,
            ..
        } => {
            let mut env = env.clone();

            for arg in args.iter().cloned().zip(arg_types.iter().cloned()) {
                env.push(arg);
            }

            if let Some(body) = body {
                typecheck_helper(&mut env, body, Some(ret_type))?;
            }

            Ok(Type::Unit)
        }

        AstNode::Return { value } => {
            if ret_type.is_none() {
                return Err("return can only occur within a function".to_string())
            }

            match value {
                Some(value) => {
                    let t = typecheck_helper(env, &**value, ret_type)?;
                    if *ret_type.unwrap() != t {
                        Err("return types do not match".to_string())
                    } else {
                        Ok(Type::Unit)
                    }
                }

                None if *ret_type.unwrap() == Type::Unit => Ok(Type::Unit),
                None => Err("return types do not match".to_string()),
            }
        }

        AstNode::Unit => Ok(Type::Unit),

        AstNode::Call { func, args } => todo!(),
    }
}

/// Performs type checking on a migrated [`Ast`] to make sure the resulting Ast is valid.
pub fn typecheck(asts: &[Ast]) -> Result<(), String> {
    let mut env = Vec::new();

    for ast in asts {
        if let AstNode::FuncDecl { name, arg_types, ret_type, .. } = &ast.ast {
            env.push((name.clone(), Type::Function(arg_types.clone(), Box::new(ret_type.clone()))));
        }
    }

    for ast in asts {
        typecheck_helper(&mut env, ast, None)?;
    }

    Ok(())
}
