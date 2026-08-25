use memmap2::MmapMut;
use std::fs::File;
use std::os::unix::io::{FromRawFd, IntoRawFd, RawFd};

/// Host-memory `memfd`/`mmap` scaffold for a future XRT/ROCm DMA-BUF bridge.
pub struct DmaBufDescriptor {
    /// Memory-mapped buffer.
    mmap: MmapMut,
    /// File descriptor for the DMA-BUF export.
    fd: Option<RawFd>,
    size: usize,
}

impl DmaBufDescriptor {
    /// Allocate a DMA-BUF of `size` bytes via memfd + mmap.
    ///
    /// In production, this would call:
    ///   - XRT xclAllocBO() for unified buffers
    ///   - dma_buf_export() via /dev/xdma ioctl
    ///   - ROCm hipImportExtBuffer() for iGPU import
    pub fn allocate(size: usize) -> anyhow::Result<Self> {
        let memfd = memfd_create("hcc-dmabuf")?;
        memfd.set_len(size.max(1) as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&memfd)? };
        let fd = Some(memfd.into_raw_fd());

        Ok(Self { mmap, fd, size })
    }

    /// Write data into the shared buffer.
    ///
    /// Errors if `data` exceeds the allocation rather than silently truncating.
    pub fn write(&mut self, data: &[u8]) -> anyhow::Result<()> {
        if data.len() > self.size {
            anyhow::bail!(
                "dmabuf payload {} bytes exceeds allocation of {} bytes",
                data.len(),
                self.size
            );
        }
        self.mmap[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from the shared buffer.
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[..self.size.min(self.mmap.len())]
    }

    /// Export the buffer as a file descriptor for cross-driver sharing.
    pub fn export_fd(&self) -> Option<RawFd> {
        self.fd
    }

    /// Size of the allocated buffer.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DmaBufDescriptor {
    fn drop(&mut self) {
        if let Some(fd) = self.fd.take() {
            unsafe {
                libc::close(fd);
            }
        }
    }
}

/// Creates an anonymous memory file descriptor (Linux memfd).
fn memfd_create(name: &str) -> std::io::Result<File> {
    let cname = std::ffi::CString::new(name).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "name contains nul byte")
    })?;
    let fd = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmabuf_alloc_write_read() {
        let mut desc = DmaBufDescriptor::allocate(4096).unwrap();
        let data = b"hello HCC DMA-BUF";
        desc.write(data).unwrap();
        assert_eq!(&desc.as_slice()[..data.len()], data);
    }

    #[test]
    fn test_dmabuf_export_fd() {
        let desc = DmaBufDescriptor::allocate(1024).unwrap();
        assert!(desc.export_fd().is_some());
    }
}
