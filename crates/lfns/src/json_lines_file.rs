//! An append-only file holding one JSON document per line.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::lfns_error::LfnsError;

/// A file of newline-separated JSON records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JsonLinesFile {
    path: PathBuf,
}

impl JsonLinesFile {
    pub(crate) fn at(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one record and flushes it to disk.
    pub(crate) fn append_record<T: Serialize>(&self, record: &T) -> Result<(), LfnsError> {
        let mut line = serde_json::to_string(record).map_err(|source| self.format_error(source))?;
        line.push('\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| self.io_error(e))?;
        file.write_all(line.as_bytes()).map_err(|e| self.io_error(e))?;
        file.sync_data().map_err(|e| self.io_error(e))
    }

    /// Reads every record; a missing file holds no records.
    pub(crate) fn read_records<T: DeserializeOwned>(&self) -> Result<Vec<T>, LfnsError> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let reader = BufReader::new(File::open(&self.path).map_err(|e| self.io_error(e))?);
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| self.io_error(e))?;
            records.push(serde_json::from_str(&line).map_err(|source| self.format_error(source))?);
        }
        Ok(records)
    }

    /// Rewrites the file so that it holds exactly `records`.
    pub(crate) fn write_records<T: Serialize>(&self, records: &[T]) -> Result<(), LfnsError> {
        let mut contents = String::new();
        for record in records {
            contents.push_str(&serde_json::to_string(record).map_err(|source| self.format_error(source))?);
            contents.push('\n');
        }
        std::fs::write(&self.path, contents).map_err(|e| self.io_error(e))
    }

    fn io_error(&self, source: std::io::Error) -> LfnsError {
        LfnsError::CheckpointIo {
            path: self.path.clone(),
            source,
        }
    }

    fn format_error(&self, source: serde_json::Error) -> LfnsError {
        LfnsError::CheckpointFormat {
            path: self.path.clone(),
            source,
        }
    }
}
