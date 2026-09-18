//! A process activity assertion for the time a request runs (macOS).
//!
//! The server has no window and nobody types into it, so after a minute or
//! two of continuous load the system stops treating its work as done on the
//! user's behalf and lets the GPU settle into a lower performance state:
//! measured on an M5 Max, the same 3-row verify pass went from 18 to 21.4 ms
//! at 1 240 MHz and 24 W where it had run at 1 620 MHz and 50 W, and
//! prefill sagged the same way, while memory-bound plain decode did not
//! notice (`docs/performance.md`, the noise section). `NSProcessInfo`'s
//! activity assertion with the user-initiated and latency-critical options
//! is the system's own way to say the opposite: it keeps the performance
//! envelope and disables timer coalescing for the process. The engine holds
//! one while a request runs and releases it in between, so an idle machine
//! can still sleep.

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

/// The assertion, held until dropped. Begun and ended on the same thread.
pub struct Activity {
    token: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

impl Activity {
    /// Begins a user-initiated, latency-critical activity with `reason`
    /// (shown by `pmset -g assertions` and Activity Monitor).
    pub fn begin(reason: &str) -> Self {
        let options =
            NSActivityOptions::UserInitiated | NSActivityOptions::LatencyCritical;
        let token = NSProcessInfo::processInfo()
            .beginActivityWithOptions_reason(options, &NSString::from_str(reason));
        Self { token }
    }
}

impl Drop for Activity {
    fn drop(&mut self) {
        // SAFETY: `token` is the object `beginActivityWithOptions_reason`
        // returned for this activity.
        unsafe { NSProcessInfo::processInfo().endActivity(&self.token) }
    }
}
