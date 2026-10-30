use cubecl_common::device::{Device, DeviceId};
use opencl_sys::{
    CL_DEVICE_EXTENSIONS, CL_DEVICE_OPENCL_C_ALL_VERSIONS, CL_DEVICE_OPENCL_C_FEATURES,
    CL_DEVICE_PLATFORM, CL_DEVICE_SVM_CAPABILITIES, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER,
    CL_DEVICE_TYPE, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU, CL_PLATFORM_NAME, cl_device_id,
    cl_device_info, cl_name_version, cl_platform_id,
};

/// A device of the `OpenCL` runtime, named by its index among the devices it can use ([`devices`]):
/// the default, 0, is the first GPU when there is one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OpenClDevice {
    pub index: usize,
}

impl OpenClDevice {
    /// How many devices the runtime can use: `OpenClDevice { index }` for every `index` below it.
    pub fn count() -> usize {
        devices().len()
    }
}

impl Device for OpenClDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id as usize,
        }
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: self.index.min(u16::MAX as usize) as u16,
        }
    }
}

/// Every device the runtime can use ([`Info::usable`]) on every `OpenCL` platform the loader lists:
/// GPUs first.
pub(crate) fn devices() -> Vec<cl_device_id> {
    if !cl3::is_opencl_runtime_available() {
        return Vec::new();
    }
    let mut devices: Vec<_> = cl3::platform::get_platform_ids()
        .unwrap_or_default()
        .into_iter()
        .flat_map(|platform| {
            cl3::device::get_device_ids(platform, CL_DEVICE_TYPE_ALL).unwrap_or_default()
        })
        .filter(|&device| Info(device).usable())
        .collect();
    devices.sort_by_key(|&device| Info(device).u64(CL_DEVICE_TYPE) & CL_DEVICE_TYPE_GPU == 0);
    devices
}

/// A device's properties.
pub(crate) struct Info(pub cl_device_id);

impl Info {
    fn data(&self, param: cl_device_info) -> Vec<u8> {
        cl3::device::get_device_data(self.0, param).unwrap_or_default()
    }

    /// A string property, empty where the device does not say.
    pub fn string(&self, param: cl_device_info) -> String {
        trimmed(&self.data(param))
    }

    /// A scalar property (up to 8 bytes), 0 where the device does not say.
    pub fn u64(&self, param: cl_device_info) -> u64 {
        word(&self.data(param))
    }

    /// A `size_t[]` property.
    pub fn sizes(&self, param: cl_device_info) -> Vec<usize> {
        self.data(param)
            .chunks_exact(size_of::<usize>())
            .map(|chunk| usize::from_le_bytes(chunk.try_into().expect("a usize")))
            .collect()
    }

    /// The device's `cl_name_version` list of `param`.
    fn names(&self, param: cl_device_info) -> Vec<(String, u32)> {
        self.data(param)
            .chunks_exact(size_of::<cl_name_version>())
            .map(|entry| {
                let version = u32::from_le_bytes(entry[..4].try_into().expect("a u32"));
                (trimmed(&entry[4..]), version)
            })
            .collect()
    }

    /// Whether the device has every extension of `names`.
    pub fn extensions(&self, names: &[&str]) -> bool {
        let extensions = self.string(CL_DEVICE_EXTENSIONS);
        names.iter().all(|name| {
            extensions
                .split_whitespace()
                .any(|extension| extension == *name)
        })
    }

    /// Whether the device's `OpenCL` C has the optional feature `name` (`OpenCL` C 3.0 feature
    /// macros).
    pub fn has_feature(&self, name: &str) -> bool {
        self.names(CL_DEVICE_OPENCL_C_FEATURES)
            .iter()
            .any(|(feature, _)| feature == name)
    }

    /// Whether the runtime can use the device: coarse-grained shared virtual memory for its buffers,
    /// and `OpenCL` C 3.0 for its kernels.
    pub fn usable(&self) -> bool {
        let svm = self.u64(CL_DEVICE_SVM_CAPABILITIES) & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER != 0;
        // CL_MAKE_VERSION(3, 0, 0)
        let opencl_c_3 = self
            .names(CL_DEVICE_OPENCL_C_ALL_VERSIONS)
            .iter()
            .any(|(_, version)| *version >> 22 >= 3);
        svm && opencl_c_3
    }

    /// The name of the device's platform.
    pub fn platform(&self) -> String {
        let platform = self.u64(CL_DEVICE_PLATFORM) as usize as cl_platform_id;
        trimmed(&cl3::platform::get_platform_data(platform, CL_PLATFORM_NAME).unwrap_or_default())
    }
}

/// A scalar of up to 8 bytes the driver returned, little-endian.
pub(crate) fn word(bytes: &[u8]) -> u64 {
    let mut word = [0u8; 8];
    let n = bytes.len().min(8);
    word[..n].copy_from_slice(&bytes[..n]);
    u64::from_le_bytes(word)
}

/// A C string from the driver, up to its NUL.
fn trimmed(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}
