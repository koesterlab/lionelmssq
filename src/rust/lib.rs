use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod rust_core {
    use pyo3::prelude::*;

    #[pyfunction]
    fn hello(name: &str) -> PyResult<String> {
        Ok(format!("Hello, {name}!"))
    }
}