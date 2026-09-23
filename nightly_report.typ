// ================================
// AO Nightly Summary Template
// ================================

// nightly_report_data.json is written by NightlyReport.SaveManifest: the
// metadata, which PNGs were produced ("figures"), the per-target summaries
// ("targets") and the files left out ("skipped"). Every section is gated on
// it, so a night with e.g. no closed-loop data never references a missing PNG.
#let data = json("nightly_report_data.json")
#let has(key) = data.figures.at(key, default: false)

// Same accent color as ao_report.typ and the plots (matplotlib's "C0").
#let accent = rgb("#1f77b4")

// See ao_report.typ: lets long underscore-joined names wrap.
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

// One per-target table cell: the median of the per-observation medians, with
// the interquartile range across observations underneath in small print.
#let summary-cell(target, key, digits: 1) = {
  let s = target.stats.at(key, default: none)
  if s == none [--] else [
    #calc.round(s.median, digits: digits) \
    #text(size: 6.5pt, fill: gray)[#calc.round(s.q25, digits: digits)–#calc.round(s.q75, digits: digits)]
  ]
}

#let mag-cell(target, band) = {
  let m = target.mags.at(band, default: none)
  if m == none [--] else [#calc.round(m, digits: 2)]
}

// A per-target table: `header` is a list of column headings (after "Target"),
// `cell` builds the remaining cells of one target's row.
#let target-table(header, cell) = {
  set text(size: 8pt)
  table(
    columns: header.len() + 1,
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
    align: (x, y) => if x == 0 { left + horizon } else { center + horizon },
    [*Target*], ..header.map(h => [*#h*]),
    ..data.targets.map(t => ([#breakable(t.name)], ..cell(t))).flatten()
  )
}

// ---------------- Title page ----------------
#align(center)[
  #if data.logo != "none" [
    #image(data.logo, height: 2cm)
    #v(10pt)
  ]
  #text(size: 20pt, weight: "bold", fill: accent)[Adaptive Optics Nightly Summary]
  #v(4pt)
  #text(size: 12.5pt, fill: gray)[#data.telescope --- Atmosphere and PSF Evolution]
  #v(16pt)
  #line(length: 35%, stroke: 1pt + accent)
  #v(16pt)
  #text(size: 15pt, weight: "bold")[#data.date]
  #v(3pt)
  #text(size: 10pt, fill: gray)[Last #data.window_hours h: #data.window_start --- #data.window_end]
]

#v(24pt)

#grid(
  columns: (auto, 1fr, auto, 1fr),
  row-gutter: 8pt,
  column-gutter: 10pt,
  [*Telescope*], [#data.telescope], [*Analyzed observations*], [#data.n_observations],
  [*Targets*], [#data.targets.len()], [*Files left out*], [#data.skipped.len()],
)

#set page(header: [
  #set text(size: 8pt, fill: gray)
  #data.telescope · Nightly summary · #data.date
  #line(length: 100%, stroke: 0.4pt + gray)
])

#if data.n_observations == 0 [
  #v(12pt)
  No analyzed observations in this window.
]

// --------------------------------
// Observed targets
// --------------------------------
#if data.targets.len() > 0 [

= Observed Targets

Per target: median of the per-observation medians; the small numbers underneath are the interquartile range across observations. SF: structure function, AC: autocorrelation, CL: closed loop, OL: open loop.

== Targets and Magnitudes
#target-table(
  ("Observations", "V", "R", "J", "H"),
  t => ([#t.n_obs], ..("V", "R", "J", "H").map(b => mag-cell(t, b))),
)

== Atmospheric Conditions
#target-table(
  ([$r_0$ telemetry (cm)], [$r_0$ PSF, CL (cm)], [$r_0$ PSF, OL (cm)], [$L_0$ (m)],
   [$tau_0$ SF (ms)], [$tau_0$ AC (ms)], [$V_0$ SF (m/s)], [$V_0$ AC (m/s)]),
  t => ("r0_wfs", "r0_psf_closed", "r0_psf_open", "L0",
        "tau0", "tau0_autocorrelation", "V0", "V0_autocorrelation").map(k => summary-cell(t, k)),
)

== Strehl Ratio and Jitter
#target-table(
  ("Strehl ratio, CL", [Jitter x, CL ($lambda slash D$)], [Jitter y, CL ($lambda slash D$)],
   [Jitter x, OL ($lambda slash D$)], [Jitter y, OL ($lambda slash D$)]),
  t => (summary-cell(t, "sr"),
        ..("jitter_x_closed", "jitter_y_closed", "jitter_x_open", "jitter_y_open").map(k => summary-cell(t, k, digits: 3))),
)

]

// --------------------------------
// Atmospheric evolution
// --------------------------------
#if has("r0") or has("L0") or has("tau0") or has("V0") [

= Atmospheric Conditions over the Night

One point per observation: median over its batches, error bar spanning the interquartile range. Target names are on the top axis.

#if has("r0") [
== Fried Parameter ($r_0$)
#figure(image("Nightly_r0.png", width: 100%), caption: [$r_0$ per observation.])
]

#if has("L0") [
== Outer Scale ($L_0$)
#figure(image("Nightly_L0.png", width: 100%), caption: [$L_0$ per observation.])
]

#if has("tau0") [
== Coherence Time ($tau_0$)
#figure(image("Nightly_tau0.png", width: 100%), caption: [$tau_0$ per observation.])
]

#if has("V0") [
== Equivalent Wind Speed ($V_0$)
#figure(image("Nightly_V0.png", width: 100%), caption: [$V_0$ per observation.])
]

]

// --------------------------------
// PSF evolution
// --------------------------------
#if has("sr") or has("jitter") [

= Strehl Ratio and Jitter over the Night

#if has("sr") [
== Strehl Ratio
#figure(image("Nightly_Strehl.png", width: 100%), caption: [Closed-loop Strehl ratio per observation.])
]

#if has("jitter") [
== Jitter
#figure(image("Nightly_Jitter.png", width: 100%), caption: [Jitter per observation.])
]

]

// --------------------------------
// Files left out
// --------------------------------
#if data.skipped.len() > 0 [

= Files Left Out

Files in the searched folders that are not part of the summary above.

#table(
  columns: 2,
  stroke: 0.5pt + gray,
  fill: (x, y) => if y == 0 { accent.lighten(85%) } else { white },
  [*File*], [*Reason*],
  ..data.skipped.map(s => ([#breakable(s.file)], [#s.reason])).flatten()
)

]

// --------------------------------
// End of document
// --------------------------------
