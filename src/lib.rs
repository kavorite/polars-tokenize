#[macro_use]
extern crate error_chain;

use polars::export::rayon::prelude::ParallelIterator;
use polars::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};
use rayon::prelude::IntoParallelRefIterator;
use std::str::FromStr;
use tokenizers::Tokenizer;

mod errors {
    error_chain! {
        foreign_links {
            ArrowSerialization(serde_arrow::Error);
            Arrow(arrow2::error::Error);
            Python(pyo3::PyErr);
            Polars(polars::error::PolarsError);
            Tokenization(tokenizers::Error);
        }
    }
    impl std::convert::From<Error> for pyo3::PyErr {
        fn from(err: Error) -> pyo3::PyErr {
            pyo3::exceptions::PyValueError::new_err(err.to_string())
        }
    }
}
use errors::*;

fn split_offsets(m: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, m)]
    } else {
        let chunk_size = m / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    m - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect()
    }
}

/// Tokenizes the input series, yielding the result as a DataFrame.
#[pyfunction]
#[pyo3(signature = (tokenizer, series, add_special_tokens=false))]
fn tokenize(tokenizer: &str, series: PySeries, add_special_tokens: bool) -> Result<PyDataFrame> {
    let series: Series = series.into();
    let tokenizer = Tokenizer::from_str(tokenizer)?;
    let series: Series = series.into();
    let offsets = split_offsets(series.len(), rayon::current_num_threads());
    let shards = offsets
        .par_iter()
        .copied()
        .map(|(offset, len)| -> Result<DataFrame> {
            let series = series.slice(offset as i64, len);
            let mut seq_ids = Vec::<u32>::new();
            let mut tok_ids = Vec::<u32>::new();
            let mut tokens = Vec::<String>::new();
            let mut attend = Vec::<u32>::new();
            for (seq_id, option) in series.utf8()?.into_iter().enumerate() {
                if let Some(value) = option {
                    let encoding = tokenizer.encode(value, add_special_tokens)?;
                    seq_ids
                        .extend(std::iter::repeat((seq_id + offset) as u32).take(encoding.len()));
                    tok_ids.extend(encoding.get_ids());
                    tokens.extend(encoding.get_tokens().into_iter().cloned());
                    attend.extend(encoding.get_attention_mask());
                }
            }
            Ok(df!(
                "attend" => attend.as_slice(),
                "seq_id" => seq_ids.as_slice(),
                "tok_id" => tok_ids.as_slice(),
                "token" => tokens.as_slice(),
            )?)
        })
        .collect::<Result<Vec<DataFrame>>>()?;
    Ok(accumulate_dataframes_vertical(shards)?).map(PyDataFrame)
}

/// A Python module implemented in Rust.
#[pymodule]
fn polars_tokenize(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
