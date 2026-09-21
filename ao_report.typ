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




#set page(
  margin: 2.5cm,
)

#set text(
  font: "Libertinus Serif",
  size: 11pt,
)

#set heading(numbering: "1.1")

#align(center)[
  #text(size: 18pt, weight: "bold")[Adaptive Optics Analysis Report]
  #v(6pt)
  #text(size: 12pt)[Atmosphere and PSF Characterization]
  #v(12pt)
  #text(size: 12pt)[Target: #target]
  #v(12pt)
  #text(size: 12pt)[#date]
]


// --------------------------------
// 1. Introduction
// --------------------------------
= Introduction

This report presents the analysis of atmospheric turbulence parameters,
control-loop characteristics, and point-spread function (PSF) diagnostics
derived from adaptive optics telemetry and image data.

// =======================================================
// 2. Observational Conditions
// =======================================================

= Observational Conditions


== Target characteristics

#columns(2)[

    - Guide star: #target
    - Elevation: #elevation deg

  #colbreak()

  ]



== AO loop characteristics

- Loop frequency: #loop_freq
- Loop gain: #loop_gain
- Loop leak: #loop_leak

// --------------------------------
// 3. Atmospheric Parameters
// --------------------------------
= Atmospheric Turbulence Parameters

== Fried Parameter ($r_0$)

#figure(
  image("AtmosphereAnalysis_r0.png", width: 80%),
  caption: [$r_0$ temporal evolution.]
)

== Coherence Time ($tau_0$)

#figure(
  image("AtmosphereAnalysis_tau0.png", width: 80%),
  caption: [$tau_0$ temporal evolution.]
)

== Equivalent Wind Speed ($V_0$)

#figure(
  image("AtmosphereAnalysis_V0.png", width: 80%),
  caption: [$V_0$ temporal evolution.]
)

// --------------------------------
// 4. Temporal and Spectral Analysis
// --------------------------------
= Temporal and Spectral Analysis

== Center-of-Gravity PSD

#figure(
  image("CoG_PSD.png", width: 85%),
  caption: [Power spectral density of the WFS center-of-gravity signals.]
)

#figure(
  image("Cumulative_Jitter.png", width: 85%),
  caption: [Cumulative jitter.]
)

#figure(
  image("PSFJitter.png", width: 85%),
  caption: [Time evolution of jitter.]
)



// --------------------------------
// 5. PSF Analysis
// --------------------------------
= PSF Analysis

== Strehl ratio

#figure(
  image("PSFAnalysis.png", width: 90%),
  caption: [Strehl ratio of long-exposure PSF.]
)

== Long exposure PSF example Frames

#figure(
  image("PSFFrames.png", width: 85%),
  caption: [Sequence of PSF frames over time.]
)

// --------------------------------
// 6. Control Loop Characterization
// --------------------------------
= Control Loop Characterization

== Loop Gain Estimation

#figure(
  image("AtmosphereAnalysis_loop_gain.png", width: 80%),
  caption: [Estimated AO loop gain.]
)

== Loop Delay Estimation

#figure(
  image("AtmosphereAnalysis_loop_delay.png", width: 80%),
  caption: [Estimated AO loop delay.]
)

// --------------------------------
// 7. Conclusions
// --------------------------------
= Conclusions

This analysis provides a comprehensive characterization of the atmospheric
conditions, AO control performance, and resulting PSF quality. These results
can be used to assess system performance and inform future optimization.

// --------------------------------
// End of document
// --------------------------------
