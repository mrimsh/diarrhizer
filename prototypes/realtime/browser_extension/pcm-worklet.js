// AudioWorklet: downmix to mono, collect `chunk` samples, post them as Int16 (the context clock rides in the message).
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunk = (options.processorOptions && options.processorOptions.chunk) || 2400;
    this.buf = new Int16Array(this.chunk);
    this.n = 0;
  }
  process(inputs) {
    const inp = inputs[0];
    if (!inp || !inp.length) return true;
    const frames = inp[0].length;
    for (let i = 0; i < frames; i++) {
      let v = 0;
      for (let c = 0; c < inp.length; c++) v += inp[c][i];
      v /= inp.length;
      this.buf[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
      if (this.n === this.chunk) {
        const out = this.buf.slice().buffer;
        this.port.postMessage({ pcm: out, t: currentTime * 1000 }, [out]);   // t: context clock (ms) at the end of the chunk
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
