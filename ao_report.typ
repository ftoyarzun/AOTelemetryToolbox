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

// One row for stat-table: two manifest keys combined as "A <sep> B unit",
// e.g. a mean +/- std pair (the default separator) or a min/max range
// (sep: " - "). `unit` is content (not a string) so a math unit like
// [$lambda/D$] renders properly.
#let stat-row(label, key-a, key-b, unit, digits: 1, sep: " ± ") = (
  label: label, key-a: key-a, key-b: key-b, unit: unit, digits: digits, sep: sep,
)

// A small "Quantity | Value" table built from stat-row(...) entries, one
// row per quantity this observation actually has data for -- entirely
// absent (renders nothing) if none of the rows do.
#let stat-table(rows) = {
  let shown = rows.filter(r => stat(r.key-a) != none)
  if shown.len() > 0 {
    table(
      columns: 2,
      stroke: 0.5pt + gray,
      fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
      [*Quantity*], [*Value*],
      ..shown.map(r => (
        [#r.label],
        [#calc.round(stat(r.key-a), digits: r.digits)#r.sep#calc.round(stat(r.key-b), digits: r.digits)#r.unit],
      )).flatten()
    )
  }
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
  [*Date*], [#date.trim("/")], [*Elevation*], [#elevation°],
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
#if has("r0") or has("L0") or has("tau0") or has("V0") [

= Atmospheric Turbulence Parameters

#if has("r0") [
== Fried Parameter ($r_0$)
#figure(
  image("AtmosphereAnalysis_r0.png", width: 80%),
  caption: [$r_0$ temporal evolution.]
)
#v(8pt)
#stat-table((
  stat-row("Telemetry", "r0_wfs_mean", "r0_wfs_std", [ cm]),
  stat-row("PSF fit, closed loop", "r0_psf_closed_mean", "r0_psf_closed_std", [ cm]),
  stat-row("PSF fit, open loop", "r0_psf_open_mean", "r0_psf_open_std", [ cm]),
  stat-row("Frozen-flow profiler", "r0_frozen_flow_mean", "r0_frozen_flow_std", [ cm]),
))
]

#if has("L0") [
== Outer Scale ($L_0$)
#figure(
  image("AtmosphereAnalysis_L0.png", width: 80%),
  caption: [$L_0$ temporal evolution.]
)
#v(8pt)
#stat-table((
  stat-row("Mean", "L0_mean", "L0_std", [ m]),
))
]

#if has("tau0") [
== Coherence Time ($tau_0$)
#figure(
  image("AtmosphereAnalysis_tau0.png", width: 80%),
  caption: [$tau_0$ temporal evolution.]
)
#v(8pt)
#stat-table((
  stat-row("Structure function", "tau0_mean", "tau0_std", [ ms]),
  stat-row("Autocorrelation", "tau0_autocorrelation_mean", "tau0_autocorrelation_std", [ ms]),
  stat-row("Frozen-flow profiler", "tau0_frozen_flow_mean", "tau0_frozen_flow_std", [ ms]),
))
]

#if has("V0") [
== Equivalent Wind Speed ($V_0$)
#figure(
  image("AtmosphereAnalysis_V0.png", width: 80%),
  caption: [$V_0$ temporal evolution.]
)
#v(8pt)
#stat-table((
  stat-row("Structure function", "V0_mean", "V0_std", [ m/s]),
  stat-row("Autocorrelation", "V0_autocorrelation_mean", "V0_autocorrelation_std", [ m/s]),
  stat-row("Frozen-flow profiler", "V0_frozen_flow_mean", "V0_frozen_flow_std", [ m/s]),
))
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
#figure(
  image("layers.png", width: 90%),
  caption: [$C_n^2$ fraction and speed per layer, and speed against direction (marker size proportional to $C_n^2$).]
)
#v(8pt)
#stat-table((
  stat-row("Layers fitted, all batches", "frozen_flow_layers_mean", "frozen_flow_layers_std", []),
  stat-row([$V_0$, all batches], "V0_frozen_flow_mean", "V0_frozen_flow_std", [ m/s]),
  stat-row([$tau_0$, all batches], "tau0_frozen_flow_mean", "tau0_frozen_flow_std", [ ms]),
))
]

// --------------------------------
// Temporal and Spectral Analysis
// --------------------------------
#if has("cog_stats") or has("jitter") [

= Temporal and Spectral Analysis

#if has("cog_stats") [
== Center-of-Gravity PSD and Cumulative Jitter
#figure(
  image("CoG_PSD.png", width: 85%),
  caption: [Power spectral density of the WFS center-of-gravity signals.]
)
#figure(
  image("Cumulative_Jitter.png", width: 85%),
  caption: [Cumulative jitter.]
)
]

#if has("jitter") [
== Jitter Time Series
#figure(
  image("PSFJitter.png", width: 85%),
  caption: [Time evolution of jitter.]
)
#v(8pt)
#stat-table((
  stat-row("x, closed loop", "jitter_x_closed_mean", "jitter_x_closed_std", [ $lambda/D$], digits: 3),
  stat-row("y, closed loop", "jitter_y_closed_mean", "jitter_y_closed_std", [ $lambda/D$], digits: 3),
  stat-row("x, open loop", "jitter_x_open_mean", "jitter_x_open_std", [ $lambda/D$], digits: 3),
  stat-row("y, open loop", "jitter_y_open_mean", "jitter_y_open_std", [ $lambda/D$], digits: 3),
))
]

]

// --------------------------------
// PSF Analysis
// --------------------------------
#if has("sr") or has("open_loop_seeing") or has("psf_frames") or has("psf_frames_openloop") [

= PSF Analysis

#if has("sr") or has("open_loop_seeing") [
== Strehl Ratio and Open-Loop Seeing
#if has("sr") [
#figure(
  image("PSFAnalysis.png", width: 90%),
  caption: [Strehl ratio of long-exposure PSF, closed loop.]
)
]
#if has("open_loop_seeing") [
#figure(
  image("PSFAnalysis_OpenLoopSeeing.png", width: 90%),
  caption: [$r_0$ from open-loop PSF fitting.]
)
]
#v(8pt)
#stat-table((
  stat-row("Strehl ratio, closed loop", "sr_mean", "sr_std", [], digits: 1),
  stat-row([$r_0$, open loop], "open_loop_seeing_r0_mean", "open_loop_seeing_r0_std", [ cm]),
))
]

#if has("psf_frames") or has("psf_frames_openloop") [
== Long-Exposure PSF Example Frames
#if has("psf_frames") [
#figure(
  image("PSFFrames.png", width: 90%),
  caption: [Closed-loop PSF frames at minimum and maximum Strehl.]
)
]
#if has("psf_frames_openloop") [
#figure(
  image("PSFFrames_OpenLoop.png", width: 90%),
  caption: [Open-loop PSF frames at worst and best $r_0$.]
)
]
#v(8pt)
#stat-table((
  stat-row("Strehl range", "sr_min", "sr_max", [], digits: 1, sep: " – "),
  stat-row([$r_0$ range, open loop], "open_loop_r0_frames_min", "open_loop_r0_frames_max", [ cm], sep: " – "),
))
]

]

// --------------------------------
// Control Loop Characterization
// --------------------------------
#if has("loop_params") or has("psd_comparison") or has("loop_bandwidth") [

= Control Loop Characterization

#if has("loop_params") [
== Loop Gain and Delay
#figure(
  image("AtmosphereAnalysis_loop_gain.png", width: 80%),
  caption: [Estimated AO loop gain.]
)
#figure(
  image("AtmosphereAnalysis_loop_delay.png", width: 80%),
  caption: [Estimated AO loop delay.]
)
#v(8pt)
#stat-table((
  stat-row("Gain", "loop_gain_mean", "loop_gain_std", [], digits: 2),
  stat-row("Delay", "loop_delay_mean", "loop_delay_std", [ frames]),
))
]

#if has("psd_comparison") [
== DM/WFS PSD Comparison
#figure(
  image("AtmosphereAnalysis_PSD_Comparison.png", width: 95%),
  caption: [DM-derived and WFS-derived power spectral densities, per mode.]
)
]

#if has("loop_bandwidth") [
== Loop Bandwidth
#figure(
  image("AtmosphereAnalysis_LoopBandwidth.png", width: 80%),
  caption: [Loop bandwidth (crossover frequency) by radial order, averaged over the observation.]
)
#v(8pt)
#stat-table((
  stat-row("Overall mean", "loop_bandwidth_mean", "loop_bandwidth_std", [ Hz], digits: 0),
))
]

]

// --------------------------------
// End of document
// --------------------------------
