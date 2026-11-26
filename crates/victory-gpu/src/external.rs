//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, moved to the host resource `state` in the
/// title list `list`, and the state it is in: `state` without the bits the
/// host copy cannot take (see `host::legal_state`). What the list records
/// next finds it there and must leave it there. The list holds the texture
/// until it is reset. `None` for a buffer, or without a host list.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    let texture = crate::resource::host_resource(resource)?.clone();
    crate::list::ensure_state(host, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        // What a library records may write buffers.
        crate::indirect::end_window(host);
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}

#[cfg(test)]
mod tests {
    use crate::object::{S_OK, guid};

    /// A host library may write buffers: recording ends a clamp window.
    #[test]
    fn library_recording_ends_a_clamp_window() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        // ID3D12CommandAllocator and ID3D12GraphicsCommandList.
        let allocator_iid = guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        let (mut allocator, mut list) = (null, null);
        assert_eq!(crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator), S_OK);
        assert_eq!(crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list), S_OK);
        let host = crate::list::host_list(list).unwrap();
        crate::indirect::open_for_test(host);
        super::record(list, |_| ()).unwrap();
        assert!(host.clamp_window().is_none());
        for object in [list, allocator] {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object);
        }
    }
}
