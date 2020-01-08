use std::thread;
use std::time;
use std::io;
use std::io::Read;
use std::f32;
use std::collections::HashMap;
use jack;

// Apparently this is idiomatic? Even though algorave is this crate?
extern crate algorave;
use algorave::backend::jack as jj;

const NUM_SINE_SAMPLES : usize = 96000;
const NUM_SINE_SAMPLES_F32 : f32 = NUM_SINE_SAMPLES as f32;
const f32_two_pi : f32 = 2.0 * f32::consts::PI;

/*
 * Static table for sine wave lookup.
 */
static mut SINE_SAMPLES : [f32; NUM_SINE_SAMPLES] = [0.0; NUM_SINE_SAMPLES];

fn fill_sine_samples() {
    let step : f32 = 1.0 / (NUM_SINE_SAMPLES as f32);
    let mut t : f32 = 0.0;
    for i in 0..NUM_SINE_SAMPLES {
        unsafe { SINE_SAMPLES[i] = f32::sin(f32_two_pi * t); }
        t += step;
    }
}

/*
 * Computes the sine wave with a given frequency, at a given sample under a
 * given sample rate: down- or up-sample to the precomputed sine wave, then
 * multiply by frequency.
 *
 * Assumes you have called fill_sine_samples.
 */
#[inline]
fn sine(n : jack::Frames, rate : jack::Frames, freq : f32, phase : f32) -> f32 {
    // Compute the point to check in the lookup table.
    //
    // First: how do we get a wave where the period is one second? We just have
    // to reconcile the sampling rates, then go modulo the precomputed rate.
    //
    // If the sampling rate is lower, the corrected sample must be higher.
    let corrected_sample = n as f32 * (NUM_SINE_SAMPLES_F32 / rate as f32);
    // Now how to include the frequency? Just multiply.
    let t = f32::floor((freq * corrected_sample + phase) % NUM_SINE_SAMPLES_F32) as usize;
    return unsafe { SINE_SAMPLES[t] };
}

#[inline]
fn sine_slow(n : jack::Frames, rate : jack::Frames, freq : f32, phase : f32) -> f32 {
    let corrected_sample = n as f32 / rate as f32;
    return f32::sin(f32_two_pi * freq * corrected_sample + phase);
}

// Having a sine lookup table may help, but it doesn't seem to be the bottleneck.
// The process callback just can't do 1500 * 128 iterations.
// Putting an empty nested loop, 128 x 1500, causes 100% CPU usage and xruns
// in JACK.

/*
 * Do a 1.0, -1.0 square cycle at this frequency.
 */
#[inline]
fn pulse(n : jack::Frames, rate : jack::Frames, freq : f32, phase : f32) -> f32 {
    return 0.0;
}

#[inline]
fn triangle(n : jack::Frames, rate : jack::Frames) -> f32 {
    return ((n % rate) as f32 / rate as f32) - 0.5;
}

fn main() {
    // Eventually we'll want the structure to be:
    //
    //   args <- parseArguments
    //   st   <- makeProgramState
    //   jack <- makeJackClient (jackArgs args) st
    //   runLangServer (langArgs args) st
    //

    fill_sine_samples();

    let client_result = jack::Client::new("algorave", jack::ClientOptions::NO_START_SERVER);
    let (client, _status) = client_result.unwrap();
    let mut controller = jj::Processor::run(client);
    controller.add_output("left", 0);
    controller.add_output("right", 1);
    controller.add_input("mic", 2);
    // The inputs and outputs appear immediately, but calling stage makes the
    // process callback become aware of them at some time in the future.
    controller.stage();
    //let async_client = client.activate_async((), processor).unwrap();

    thread::sleep(time::Duration::from_millis(100000000));

    // TODO port addition should be dynamic, controlled by the programming
    // language.
    //let port_left = async_client.as_client().register_port("left", jack::AudioOut);
    //let port_right = async_client.as_client().register_port("right", jack::AudioOut);
    //let port_left = client.register_port("left", jack::AudioOut).unwrap();
    //let _port_right = client.register_port("right", jack::AudioOut).unwrap();
    //let mut ports = HashMap::new();
    //ports.insert("left", jj::Output::JackAudioOutput(port_left));
    // ports.insert("right", Output::JackAudioOutput(port_right));

    //let ten_s = time::Duration::from_millis(10000);
    //thread::sleep(ten_s);
    /*let mut buf = [0; 64];
    let n = std::io::stdin().read(&mut buf).unwrap();
    println!("Echo: {:?}", &buf[..n]);
    */

}
