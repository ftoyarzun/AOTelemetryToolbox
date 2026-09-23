// ================================
// AO Analysis Report Template
// ================================

// ---------------- Metadata ----------------
#let report-title = sys.inputs.AOtitle
#let telescope    = sys.inputs.telescope
#let date         = sys.inputs.date
#let target       = sys.inputs.target
#let elevation    = sys.inputs.elevation
#let loop_gain    = sys.inputs.loop_gain
#let loop_leak    = sys.inputs.loop_leak
#let loop_freq    = sys.inputs.loop_freq
#let VMag         = sys.inputs.at("VMag", default: "none")
#let RMag         = sys.inputs.at("RMag", default: "none")
#let JMag         = sys.inputs.at("JMag", default: "none")
#let HMag         = sys.inputs.at("HMag", default: "none")
#let logo         = sys.inputs.at("logo", default: "none")

// report_data.json is written by AnalysisViewer.SaveFigureManifest --
// "figures" says which PNGs actually got produced for this observation
// (some regimes -- closed loop, open loop -- may be entirely absent), and
// "stats" holds small mean/std summaries for each figure's table. Gating
// every section on this instead of assuming every PNG exists is what lets
// one template serve mixed, closed-only and open-only observations without
// ever referencing a file that was never written.
#let manifest = json("report_data.json")
#let has(key) = manifest.figures.at(key, default: false)
#let stat(key) = manifest.stats.at(key, default: none)

// One accent color, shared with the plots (matplotlib's default "C0" blue),
// so the template and the figures it wraps read as one coherent report.
#let accent = rgb("#1f77b4")

// Target names and observation IDs are often one long underscore-joined
// token (e.g. a simulated-data filename) with no spaces or hyphens --
// Typst's line breaking doesn't treat "_" as a break opportunity, so a long
// one overflows its container instead of wrapping. A zero-width space after
// each underscore fixes that without changing how the text looks.
#let breakable(s) = str(s).replace("_", "_\u{200B}")

// ---------------- Page & text setup ----------------
#set page(margin: 2.5cm, numbering: "1 / 1")
#set text(font: "Libertinus Serif", size: 11pt)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  line(length: 100%, stroke: 1.2pt + accent)
  v(6pt)
  set text(size: 16pt, weight: "bold", fill: accent)
  it
  v(4pt)
}

#show heading.where(level: 2): it => {
  v(10pt, weak: true)
  set text(size: 12.5pt, weight: "bold", fill: accent.darken(15%))
  it
  v(4pt, weak: true)
}

// Analysis settings the captions quote (batch lengths), from the manifest
#let setting(key) = manifest.at("settings", default: (:)).at(key, default: none)

// `x` with exactly `digits` decimals, trailing zeros kept.
#let fixed(x, digits) = {
  let s = str(calc.round(x, digits: digits))
  if digits == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (digits - frac.len())
}

// Decimals that show a standard deviation with 2 significant figures when
// its leading digit is 1 or 2, else 1.
#let std-digits(std) = {
  let e = calc.floor(calc.log(std))
  let sig = if std / calc.pow(10.0, e) < 3 { 2 } else { 1 }
  calc.max(0, sig - 1 - e)
}

// Rows for stat-table. stat-row: the "<key>_mean", "<key>_std" and "<key>_n"
// manifest entries, as "mean ± std unit", the std rounded by std-digits and
// the mean to the same decimal; `digits` is used when the std is 0.
// range-row: two manifest keys as "min – max unit", with the batch count
// from `key-n`. `unit` is content (not a string) so a math unit like
// [$lambda/D$] renders properly.
#let stat-row(label, key, unit, digits: 1) = (
  label: label, kind: "mean", key: key, unit: unit, digits: digits,
)
#let range-row(label, key-min, key-max, key-n, unit, digits: 1) = (
  label: label, kind: "range", key-min: key-min, key-max: key-max, key-n: key-n, unit: unit, digits: digits,
)

#let row-value(r) = {
  if r.kind == "range" {
    [#fixed(stat(r.key-min), r.digits) – #fixed(stat(r.key-max), r.digits)#r.unit]
  } else {
    let mean = stat(r.key + "_mean")
    let std = stat(r.key + "_std")
    let n = stat(r.key + "_n")
    if n == 1 or std == none {
      [#fixed(mean, r.digits)#r.unit]
    } else {
      let d = if std > 0 { std-digits(std) } else { r.digits }
      [#fixed(mean, d) ± #fixed(std, d)#r.unit]
    }
  }
}

// A small "Quantity | Value | Batches" table built from stat-row(...) and
// range-row(...) entries, one row per quantity this observation actually has
// data for -- entirely absent (renders nothing) if none of the rows do.
#let stat-table(rows) = {
  let first-key(r) = if r.kind == "range" { r.key-min } else { r.key + "_mean" }
  let n-key(r) = if r.kind == "range" { r.key-n } else { r.key + "_n" }
  let shown = rows.filter(r => stat(first-key(r)) != none)
  if shown.len() > 0 {
    table(
      columns: 3,
      stroke: 0.5pt + gray,
      fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
      [*Quantity*], [*Value*], [*Batches*],
      ..shown.map(r => (
        [#r.label],
        row-value(r),
        [#if stat(n-key(r)) == none [--] else [#stat(n-key(r))]],
      )).flatten()
    )
  }
}

// A sentence giving the length of the batches of `key` in the settings.
#let batch-length(key, what) = {
  let b = setting(key)
  if b != none [ #what #b s long.]
}

// ---------------- Title page ----------------
#align(center)[
  #if logo != "none" [
    #image(logo, height: 2cm)
    #v(10pt)
  ]
  #text(size: 20pt, weight: "bold", fill: accent)[Adaptive Optics Analysis Report]
  #v(4pt)
  #text(size: 12.5pt, fill: gray)[#telescope --- Atmosphere and PSF Characterization]
  #v(16pt)
  #line(length: 35%, stroke: 1pt + accent)
  #v(16pt)
  #text(size: 15pt, weight: "bold")[#breakable(target)]
  #v(3pt)
  #text(size: 11pt, fill: gray)[#date.trim("/")]
  #v(3pt)
  #text(size: 9pt, fill: gray)[Observation: #breakable(report-title)]
]

#v(24pt)

#text(size: 12.5pt, weight: "bold", fill: accent)[Observational Conditions]
#v(6pt)

#grid(
  columns: (auto, 1fr, auto, 1fr),
  row-gutter: 8pt,
  column-gutter: 10pt,
  [*Telescope*], [#telescope], [*Target*], [#breakable(target)],
  [*Date (UTC)*], [#date.trim("/")], [*Elevation*], [#elevation°],
)

#v(10pt)
#table(
  columns: 3,
  stroke: 0.5pt + gray,
  fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
  align: center,
  [*Loop gain*], [*Loop leak*], [*Loop frequency (Hz)*],
  [#loop_gain], [#loop_leak], [#loop_freq],
)

#if VMag != "none" [
  #v(10pt)
  #table(
    columns: 4,
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
    align: center,
    [*V*], [*R*], [*J*], [*H*],
    [#VMag], [#RMag], [#JMag], [#HMag],
  )
]

#pagebreak()
#outline(title: [Contents])

#set page(header: [
  #set text(size: 8pt, fill: gray)
  #telescope · #breakable(target) · #date.trim("/")
  #line(length: 100%, stroke: 0.4pt + gray)
])

// --------------------------------
// Atmospheric Parameters
// --------------------------------
#if has("r0") or has("tau0") or has("V0") [

= Atmospheric Turbulence Parameters

Along the line of sight, at the telescope elevation. One point per batch; the tables give the mean and standard deviation over the batches. Telemetry batches overlap by half.#batch-length("atmosphere_batch_s", [They are]) PSF batches do not overlap.#batch-length("psf_batch_s", [They are])#if setting("frozen_flow_batch_frames") != none [ Frozen-flow batches are up to #setting("frozen_flow_batch_frames") loop frames long.]

#if has("r0") [
== Fried Parameter ($r_0$)
#block(breakable: false)[
#figure(
  image("AtmosphereAnalysis_r0.png", width: 80%),
  caption: [$r_0$ per batch, from the DM commands (telemetry and frozen-flow profiler) and from the fit of the long-exposure PSF.]
)
#v(8pt)
#stat-table((
  stat-row("Telemetry", "r0_wfs", [ cm]),
  stat-row("PSF fit, closed loop", "r0_psf_closed", [ cm]),
  stat-row("PSF fit, open loop", "r0_psf_open", [ cm]),
  stat-row("Frozen-flow profiler", "r0_frozen_flow", [ cm]),
))
]
]

#if has("tau0") [
== Coherence Time ($tau_0$)
#block(breakable: false)[
#figure(
  image("AtmosphereAnalysis_tau0.png", width: 80%),
  caption: [$tau_0$ per batch, from the frozen-flow profiler and from the temporal autocorrelation of the DM commands.]
)
#v(8pt)
#stat-table((
  stat-row("Frozen-flow profiler", "tau0_frozen_flow", [ ms]),
  stat-row("Autocorrelation (cross-check)", "tau0_autocorrelation", [ ms]),
))
]
]

#if has("V0") [
== Equivalent Wind Speed ($V_0$)
#block(breakable: false)[
#figure(
  image("AtmosphereAnalysis_V0.png", width: 80%),
  caption: [$V_0$ per batch, from the frozen-flow profiler and from the temporal autocorrelation of the DM commands.]
)
#v(8pt)
#stat-table((
  stat-row("Frozen-flow profiler", "V0_frozen_flow", [ m/s]),
  stat-row("Autocorrelation (cross-check)", "V0_autocorrelation", [ m/s]),
))
]
]

]

// --------------------------------
// Frozen-Flow Layers
// --------------------------------
#if has("frozen_flow") [

= Frozen-Flow Layers

Multi-layer frozen-flow fit of the DM-command slope autocorrelation, first closed-loop batch. Velocities and directions are in the axes of the DM actuator grid.

== Correlation Cube
#figure(
  image("correlation.png", width: 95%),
  caption: [Correlation cube (data), multi-layer model and residual at a few time lags. Crosses: position of each fitted layer at that lag.]
)

== Layer Maps
#figure(
  image("layer_maps.png", width: 100%),
  caption: [Fitted 2D correlation map of each layer.]
)

== Layer Strengths, Speeds and Directions
#block(breakable: false)[
#figure(
  image("layers.png", width: 90%),
  caption: [$C_n^2$ fraction and speed per layer, and speed against direction (marker size proportional to $C_n^2$).]
)
#v(8pt)
#stat-table((
  stat-row("Layers fitted, all batches", "frozen_flow_layers", []),
  stat-row([$V_0$, all batches], "V0_frozen_flow", [ m/s]),
  stat-row([$tau_0$, all batches], "tau0_frozen_flow", [ ms]),
))
]
]

// --------------------------------
// Temporal and Spectral Analysis
// --------------------------------
#if has("cog_stats") or has("jitter") [

= Temporal and Spectral Analysis

#if has("cog_stats") [
== Center-of-Gravity PSD and Cumulative Jitter
#block(breakable: false)[
#figure(
  image("CoG_PSD.png", width: 85%),
  caption: [Power spectral density of the center of gravity of the science-camera PSF, over the longest closed-loop and the longest open-loop run.]
)
]
#block(breakable: false)[
#figure(
  image("Cumulative_Jitter.png", width: 85%),
  caption: [Cumulative jitter: square root of the center-of-gravity PSD integrated up to each frequency.]
)
]
]

#if has("jitter") [
== Jitter Time Series
#block(breakable: false)[
#figure(
  image("PSFJitter.png", width: 85%),
  caption: [Jitter per batch: standard deviation of the center of gravity of the science-camera PSF within each batch.#batch-length("psf_batch_s", [Batches are])]
)
#v(8pt)
#stat-table((
  stat-row("x, closed loop", "jitter_x_closed", [ $lambda slash D$], digits: 3),
  stat-row("y, closed loop", "jitter_y_closed", [ $lambda slash D$], digits: 3),
  stat-row("x, open loop", "jitter_x_open", [ $lambda slash D$], digits: 3),
  stat-row("y, open loop", "jitter_y_open", [ $lambda slash D$], digits: 3),
))
]
]

]

// --------------------------------
// PSF Analysis
// --------------------------------
#if has("sr") or has("open_loop_seeing") or has("psf_frames") or has("psf_frames_openloop") [

= PSF Analysis

#if has("sr") or has("open_loop_seeing") [
== Strehl Ratio and Open-Loop Seeing
#block(breakable: false)[
#if has("sr") [
#figure(
  image("PSFAnalysis.png", width: 90%),
  caption: [Strehl ratio of the long-exposure PSF of each closed-loop batch, from the PSF model fit.#batch-length("psf_batch_s", [Batches are])]
)
]
#if has("open_loop_seeing") [
#figure(
  image("PSFAnalysis_OpenLoopSeeing.png", width: 90%),
  caption: [$r_0$ from the fit of the long-exposure PSF of each open-loop batch.]
)
]
#v(8pt)
#stat-table((
  stat-row("Strehl ratio, closed loop", "sr", [ %]),
  stat-row([$r_0$, open loop], "open_loop_seeing_r0", [ cm]),
))
]
]

#if has("psf_frames") or has("psf_frames_openloop") [
== Long-Exposure PSF Example Frames
#block(breakable: false)[
#if has("psf_frames") [
#figure(
  image("PSFFrames.png", width: 90%),
  caption: [Closed-loop long-exposure PSFs at minimum and maximum Strehl ratio: data, fit and fit − data.]
)
]
#if has("psf_frames_openloop") [
#figure(
  image("PSFFrames_OpenLoop.png", width: 90%),
  caption: [Open-loop long-exposure PSFs at smallest and largest $r_0$: data, fit and fit − data.]
)
]
#v(8pt)
#stat-table((
  range-row("Strehl ratio range", "sr_min", "sr_max", "sr_n", [ %]),
  range-row([$r_0$ range, open loop], "open_loop_r0_frames_min", "open_loop_r0_frames_max", "open_loop_seeing_r0_n", [ cm]),
))
]
]

]

// --------------------------------
// Control Loop Characterization
// --------------------------------
#if has("psd_comparison") [

= Control Loop Characterization

== DM/WFS PSD Comparison
#block(breakable: false)[
#figure(
  image("AtmosphereAnalysis_PSD_Comparison.png", width: 95%),
  caption: [Temporal PSD per mode of the DM commands and of the WFS measurements (closed-loop residual) in the last closed-loop batch, and of the WFS measurements in the last open-loop batch.]
)
]

]

// --------------------------------
// End of document
// --------------------------------
