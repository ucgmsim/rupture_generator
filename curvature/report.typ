#import "@preview/cetz:0.4.2"

#let cool = rgb("#17527D")
#let warm = rgb("#A93A26")
#let muted = rgb("#58626F")

#let control = [#text(fill: cool, weight: "bold", size: 0.8em, tracking: 0.08em)[CONTROL]]
#let treatment = [#text(fill: warm, weight: "bold", size: 0.8em, tracking: 0.08em)[TREATMENT]]

= What a planar fault costs on the Hikurangi interface

A subduction interface is not a plane. The usual treatment in the literature is to
generate a rupture on a plane fitted to the interface and project the result onto the
real geometry. This study measures what that costs, at Mw 8.5 on the NZ CFM v1.0
Hikurangi interface, discretised at 525 m.

The short answer is that almost all of the cost is *depth* rather than the arc-length
stretch the projection is usually discussed in terms of; that almost all of the depth cost
is *fixable* without abandoning the plane; and that what survives is small enough to sit
inside the other uncertainties in a source model. This is a more deflationary conclusion
than the study set out expecting, and it is worth stating plainly at the top.

Fitting a plane to a surface descending from 3 km to 75 km puts each subfault at the wrong
depth, and depth sets both the rigidity that converts slip to moment and the shear speed
that carries the rupture front. On Hikurangi that costs a median
#text(font: "DejaVu Sans Mono")[7.5] s of rupture timing — two orders of magnitude more
than the geometric term. But reading material properties at the *true* depth while leaving
the geometry flat removes essentially all of it, on every interface and from every
hypocentre tested, leaving a residual of #text(font: "DejaVu Sans Mono")[0.1]–#text(font: "DejaVu Sans Mono")[0.3] s.

What cannot be fixed that way is the area: the projection loses surface, so slip scaled to
target on planar areas over-delivers moment by #text(font: "DejaVu Sans Mono")[3.3%] on
Hikurangi and #text(font: "DejaVu Sans Mono")[3.4]–#text(font: "DejaVu Sans Mono")[6.8%] on
the Puysegur interfaces. Against magnitude uncertainty and scaling-relation scatter that is
not a large number, and this is the least favourable case anyone would construct — a
*single* plane fitted to an entire interface, where a real scenario ruptures a portion and
gets a far better-fitting plane.

So the honest reading is that a planar interface is defensible for moment, provided
materials are read at true depth. What it costs is timing if you do not make that
correction, and a geometry that places #text(font: "DejaVu Sans Mono")[8.5%] of the
Hikurangi interface above sea level — which is wrong in kind rather than in degree, and is
the one result here that no error budget absorbs.

== The comparison

Both models are the same Monge patch. A reference plane is fitted to the interface by
singular value decomposition — the least-squares plane, which minimises the out-of-plane
displacement $h$ by construction and is therefore the most generous possible reading of
an approach whose papers rarely say how the projection plane was built. Every vertex has
parameter coordinates $(u, v)$ on that plane. The two models differ in one term.

#html.frame(cetz.canvas(length: 1cm, {
  import cetz.draw: *
  set-style(stroke: (thickness: 0.5pt))

  // the reference plane, seen edge-on
  line((0, 0), (11, 0), stroke: (paint: cool, thickness: 1pt))
  content((11.6, 0), text(size: 8pt, fill: cool)[plane])

  // the true interface: a descending, undulating profile
  let prof = ((0, 0.9), (1.5, 0.75), (3, 0.3), (4.5, -0.35), (6, -0.75), (7.5, -0.6), (9, -0.1), (10.5, 0.55))
  merge-path(stroke: (paint: warm, thickness: 1.2pt), {
    hobby(..prof)
  })
  content((11.7, 0.6), text(size: 8pt, fill: warm)[interface])

  // h at two sample stations, one either side
  line((3, 0), (3, 0.3), stroke: (paint: muted, dash: "dotted"))
  line((6, 0), (6, -0.75), stroke: (paint: muted, dash: "dotted"))
  content((3.35, 0.17), text(size: 7pt, fill: muted)[$h > 0$])
  content((6.4, -0.4), text(size: 7pt, fill: muted)[$h < 0$])

  circle((3, 0), radius: 0.07, fill: cool, stroke: none)
  circle((3, 0.3), radius: 0.07, fill: warm, stroke: none)
  circle((6, 0), radius: 0.07, fill: cool, stroke: none)
  circle((6, -0.75), radius: 0.07, fill: warm, stroke: none)

  // the normal
  line((8.2, 0), (8.2, 0.8), stroke: (paint: cool), mark: (end: ">", scale: 0.5))
  content((8.55, 0.85), text(size: 7pt, fill: cool)[$hat(n)$])
}))

#html.elem("p", attrs: (class: "note"))[
  Section through the patch. The plane is fitted to the interface, so $h$ changes sign
  across it and the two models straddle rather than offset one another.
]

The curved model places each vertex at $X = O + u hat(e)_u + v hat(e)_v + h hat(n)$ —
the real interface. The flat model drops the last term. Everything downstream then
follows from geometry: cell areas, the distances the eikonal solver integrates along,
the depth at which the velocity model is sampled, and the metric the correlated slip
field is drawn against.

On Hikurangi the plane's normal is #text(font: "DejaVu Sans Mono")[14.11°] off vertical,
so $h$ is very nearly a depth error. It runs from #text(font: "DejaVu Sans Mono")[−24.9]
to #text(font: "DejaVu Sans Mono")[+16.3] km.

== Method

The two models share a mesh topology. The flat twin has *identical* faces and *identical*
parameter coordinates, so it has the same number of degrees of freedom, and the sampler
draws the *same white-noise vector* from the same seed in both. Every difference reported
here is geometry. Nothing is a different random field.

#table(
  columns: (auto, auto, 1fr),
  stroke: none,
  table.hline(),
  table.header(
    [*hypocentre*], [*velocity model*], [*what it isolates*],
  ),
  table.hline(),
  [central], [constant], [#control geometry alone — area, path length, metric],
  [central], [standard 1-D], [#treatment geometry #sym.plus depth],
  [northern], [standard 1-D], [spatial variability],
  [southern], [standard 1-D], [spatial variability],
  table.hline(),
)

The first two rows are the controlled pair: same hypocentre, same noise, same geometry
pair, and only the velocity model changes. Their difference *is* the depth contribution,
measured rather than argued. For the control to be a control, constant shear speed is not
enough — density is held constant too, so rigidity is a single number, and the depth ramps
that reduce rupture speed and set rise time are flattened, so the eikonal sees a uniform
slowness field.

Hypocentres sit 180 km down dip at
#text(font: "DejaVu Sans Mono")[177.853°E, −38.514°] (offshore Gisborne),
#text(font: "DejaVu Sans Mono")[176.167°E, −40.316°] (south Hawke's Bay), and
#text(font: "DejaVu Sans Mono")[174.386°E, −42.094°] (offshore Marlborough), at true
depths of 19.4, 19.6 and 16.0 km.

== The result

#html.elem("div", attrs: (class: "numeric"))[
  #table(
    columns: (1fr, auto, auto),
    stroke: none,
    align: (left, right, right),
    table.hline(),
    table.header([], [#control *geometry alone*], [#treatment *geometry + depth*]),
    table.hline(),
    [onset error, median], [−0.075 s], [*+7.53 s*],
    [onset error, p90], [−0.009 s], [+21.9 s],
    [onset error, largest], [1.26 s], [*34.3 s*],
    [faces arriving early], [99.99993%], [31.4%],
    [moment delivered / target], [1.0258], [0.9690],
    [duration, flat / curved], [0.998], [1.024],
    table.hline(),
  )
]

Depth beats arc length by roughly *two orders of magnitude* on onset and by a factor of
2.4 on moment, and it reverses the sign of both.

Geometrically the flat model can only arrive early: projecting onto a plane shortens every
path. It does so on 99.99993% of faces, by a median of 75 milliseconds — negligible
against a rupture lasting some 190 s. Introduce the depth-dependent velocity model and the
flat model arrives *late* by 7.5 s at the median and 34 s at worst, which is 19% of the
duration. The two mechanisms pull in opposite directions and depth wins outright.

#figure(
  image("figures/onset_polar_control.png"),
  caption: [#control Onset difference under constant velocity, as a function of azimuth
    and distance from the hypocentre. Pure geometry: uniformly, negligibly early.],
)

#figure(
  image("figures/onset_polar.png"),
  caption: [#treatment The same plot with the standard velocity model. The quadrupole is
    the mechanism — late up-dip and along strike, early down-dip — and it is the depth
    error rather than the path length that produces it.],
)

== What the plane discards

The driver is the depth error $Delta z = z_"flat" - z_"curved"$, which runs from
#text(font: "DejaVu Sans Mono")[−24.0] to #text(font: "DejaVu Sans Mono")[+15.8] km with
a median absolute value of 6.2 km. Set that against the surface-to-projected area ratio,
which is only #text(font: "DejaVu Sans Mono")[1.0281]: the depth error is an order of
magnitude larger in its own units and it enters through more channels.

#figure(
  image("figures/depth_error.png"),
  caption: [The depth error across the interface. Because the plane is a least-squares
    fit it passes through the middle of the surface, so the error changes sign and
    partially cancels in totals while remaining large everywhere locally.],
)

Three consequences follow, and none is subtle:

- *38.2% of faces are given the wrong rigidity* — 530,686 of 1,389,600 — with the true
  value up to #text(font: "DejaVu Sans Mono")[2.33]× the flat model's.
- *41.8% land in a different velocity layer*, by as much as 13 layers.
- *8.5% of the interface is pushed above sea level.* That is 14,943 km², with the flat
  depth reaching #text(font: "DejaVu Sans Mono")[−17.6] km. The plane does not merely
  misplace this part of the fault; it puts it in the air.

== Moment

Moment is the product $mu A s$, and the flat model gets two of the three factors wrong in
opposite directions. Because the errors are multiplicative they separate cleanly: the
interaction term is #text(font: "DejaVu Sans Mono")[1.0014], so the decomposition is
exact to within a seventh of a percent.

#html.elem("div", attrs: (class: "numeric"))[
  #table(
    columns: (1fr, auto),
    stroke: none,
    align: (left, right),
    table.hline(),
    [area — the projection loses surface], [*+3.25%*],
    [rigidity — subfaults sit at the wrong depth], [*−6.16%*],
    table.hline(stroke: 0.4pt),
    [net], [*−3.10%*],
    table.hline(),
  )
]

The rigidity error is the larger of the two, and it happens to oppose the area error, so
the delivered magnitude is Mw #text(font: "DejaVu Sans Mono")[8.491] against
#text(font: "DejaVu Sans Mono")[8.5] asked. A modeller checking only the total moment
would see a 0.9% discrepancy and conclude the projection was almost harmless. It is not:
the cancellation is a coincidence of this interface and this magnitude, and the
per-subfault errors that produced it are large.

Note what the moment rescaling does here. The generator scales slip so the total moment
hits its target, and the flat model scales against *planar* areas. Projecting that slip
onto the real, larger surface is therefore what causes the error — the rescaling does not
absorb it, it creates it.

#figure(
  image("figures/moment.png"),
  caption: [Moment attribution. The area and rigidity contributions are separable
    because the errors are multiplicative.],
)

== Slip correlation

Measured against true surface separation, the curved model delivers correlation lengths of
#text(font: "DejaVu Sans Mono")[42.6] km along strike and
#text(font: "DejaVu Sans Mono")[23.1] km down dip. The flat field projected onto the
interface delivers #text(font: "DejaVu Sans Mono")[46.3] and
#text(font: "DejaVu Sans Mono")[24.1] km — *8.7% and 4.0% too long*.

Only about 0.5–0.9% of that comes from the projection stretch itself. The remainder is
that the Laplace–Beltrami operator on the real surface delivers a shorter length than the
same operator on the plane: the correlation structure is a property of the surface's
metric, not of a distance computed on its shadow.

One honest caveat: *both* models under-deliver along strike, at 0.76–0.82 of the length
asked for. That is a shared estimator and boundary bias, so the ratio between the models
is the finding here and the absolute figures are not.

The two slip fields correlate pointwise at #text(font: "DejaVu Sans Mono")[0.988] — a
number that only means something because the two draws share their white noise.

#figure(
  image("figures/correlation.png"),
  caption: [Empirical correlation against true surface separation, along strike and down
    dip, for both models.],
)

== Moment rate function

#figure(
  image("figures/moment_rate.png"),
  caption: [Moment rate functions and their amplitude spectra. Under constant velocity the
    two are indistinguishable; with depth they separate.],
)

Under the control the two spectra are *indistinguishable* — corner frequency
#text(font: "DejaVu Sans Mono")[0.00481] against
#text(font: "DejaVu Sans Mono")[0.00484] Hz, and a high-frequency falloff of
#text(font: "DejaVu Sans Mono")[−2.12] in both. This is a genuine null result and worth
stating as one: the geometric stretch alone does not change the radiated spectrum.

With the depth-dependent model they separate. Peak moment rate falls
#text(font: "DejaVu Sans Mono")[6.3%], the corner frequency drops from
#text(font: "DejaVu Sans Mono")[0.00432] to #text(font: "DejaVu Sans Mono")[0.00369] Hz —
a 15% shift — and the falloff steepens from #text(font: "DejaVu Sans Mono")[−1.85] to
#text(font: "DejaVu Sans Mono")[−1.95]. A rupture on the plane radiates as a slightly
larger, slower event than the same rupture on the real interface.

The spectra are trustworthy to about 6 Hz, where the 525 m subfaults stop resolving the
front. The sample interval of 0.02 s gives a Nyquist of 25 Hz, so the discretisation
rather than the sampling sets that limit.

== Plan and section views

#figure(
  image("figures/plan_view.png"),
  caption: [Plan view. The interface, the fitted plane, and where they part company.],
)

#figure(
  image("figures/sections.png"),
  caption: [Down-dip sections at three along-strike positions, chosen for the largest
    deviation, a typical one, and a near-planar one. The plane crosses the interface
    rather than sitting beside it, which is why $Delta z$ changes sign.],
)

#figure(
  image("figures/slip_maps.png"),
  caption: [Slip, both models, and their difference. The fields share their white noise,
    so the difference is geometry.],
)

#figure(
  image("figures/onset_maps.png"),
  caption: [Onset, both models, and their difference.],
)

== Spatial variability

The central hypocentre gives the *smallest* effect of the three. Median onset error is
+11.3 s at the northern site, reaching 45.7 s, and +11.8 s at the southern site, reaching
48.2 s. The headline numbers above are therefore conservative.

== What survives standardisation

The generator standardises each drawn field to zero mean and unit sample variance, which
matters for reading these results. The 2.0% difference in the drawn field's sample spread
*does not survive* — it is divided out and never reaches the output. The structural
difference survives exactly: Pearson correlation is affine-invariant, so the two fields
correlate at #text(font: "DejaVu Sans Mono")[0.98672] both before and after.

What reaches an SRF is therefore a −3.1% amplitude error alongside a 7.5 s onset error,
not the variance difference.

== Resolution

#figure(
  image("figures/resolution.png"),
  caption: [Area ratio against mesh spacing. Subdivision is exactly invariant; rebuilding
    converges.],
)

The geometry is band-limited by the source. The CFM interface has a median vertex spacing
of 5.6 km and the mesh builder lifts by piecewise-linear interpolation on the source
faces, so a mesh finer than the source produces coplanar sub-triangles and inherits the
source's own $|gradient h|$ and area exactly. At 525 m the geometry is oversampled
elevenfold, and every geometric quantity here is what a 400 m mesh would give.

Two distinct behaviours are worth separating. *Subdivision* is exactly invariant — the
area ratio is identical to twelve digits across an eightfold refinement, because midpoint
refinement puts new vertices on the parent faces. *Rebuilding* is not: the ratio converges
from #text(font: "DejaVu Sans Mono")[1.026248] at 8 km to
#text(font: "DejaVu Sans Mono")[1.028638] at 500 m, approaching
#text(font: "DejaVu Sans Mono")[1.0287]. It is the base spacing, not the number of
refinements, that sets geometric fidelity.

== The counterfactual: correct materials on the wrong geometry

If depth is what does the damage, then assigning material properties from the *mesh
geometry* rather than from the rupture sampler should let the projection compete squarely.
A third condition tests it: flat geometry throughout — the plane's shorter paths, its
areas, its metric for the correlated field — but rigidity, shear speed and every
depth-dependent ramp read at the *true* interface depth.

It works, and it very nearly closes the gap.

#html.elem("div", attrs: (class: "numeric"))[
  #table(
    columns: (1fr, auto, auto, auto),
    stroke: none,
    align: (left, right, right, right),
    table.hline(),
    table.header(
      [], [*status quo*], [*true-depth*], [#control *geometry alone*],
    ),
    table.hline(),
    [onset error, median], [+7.53 s], [*−0.174 s*], [−0.075 s],
    [onset error, p90], [+21.9 s], [*−0.011 s*], [−0.009 s],
    [onset error, largest], [34.3 s], [*4.28 s*], [1.26 s],
    [moment delivered / target], [0.9690], [*1.0325*], [1.0258],
    table.hline(),
  )
]

The refactor removes #text(font: "DejaVu Sans Mono")[102%] of the onset error. The
true-depth column collapses onto the constant-velocity control, which is the pure-geometry
floor — confirming from the other direction that essentially all of the 7.5 s was depth.

#figure(
  image("figures/true_depth.png"),
  caption: [The three conditions against the curved model. Correcting the depth at which
    materials are read removes almost the whole onset error.],
)

*But it has to be done completely.* A partial version — correcting rigidity and shear
speed while leaving the three depth ramps reading planar depths — gives a median onset
error of #text(font: "DejaVu Sans Mono")[+9.10 s], which is *worse than doing nothing*.
Half the refactor is worse than none of it, because the ramps and the velocity sampling
then disagree about where the subfault is.

What the working version reads at true depth: rigidity and shear speed in
`sample_velocity_model`, the rupture-speed ramp in `timing`, both rise-time ramps, and the
pulse's rising fraction. What stays flat: the vertices the front propagates over, the areas
the moment folds over, the SPDE operator's metric, and the lateral taper.

=== The irreducible cost

Moment is where the refactor stops helping. It moves from
#text(font: "DejaVu Sans Mono")[0.9690] to #text(font: "DejaVu Sans Mono")[1.0325] — and
that second number is *exactly* the area contribution measured earlier. Correcting the
depths removes the rigidity error and leaves the geometry alone, which is the whole point;
but it is no better in magnitude, because the status quo's smaller net figure was the
accidental cancellation between the two.

So the irreducible geometric cost, once materials are correct, is a *negligible* onset
error and a *real* #text(font: "DejaVu Sans Mono")[+3.25%] moment error. That one cannot be
fixed by reading materials better. It needs the true surface areas.

#figure(
  image("figures/true_depth_moment.png"),
  caption: [Moment under the three conditions. Correcting materials exposes the area error
    rather than removing it.],
)

=== The equivalence holds everywhere

The constant-velocity control and the true-depth counterfactual are two different ways of
asking the same question, and they agree — at every hypocentre, on all three interfaces.
Median onset error, in seconds:

#html.elem("div", attrs: (class: "numeric"))[
  #table(
    columns: (auto, auto, auto, auto, auto),
    stroke: none,
    align: (left, left, right, right, right),
    table.hline(),
    table.header(
      [*interface*], [*site*], [#control *geometry*], [*true-depth*], [*status quo*],
    ),
    table.hline(),
    [Hikurangi], [northern], [−0.079], [−0.137], [*+11.283*],
    [], [central], [−0.075], [−0.174], [*+7.531*],
    [], [southern], [−0.090], [−0.156], [*+11.830*],
    [Puysegur–Fiordland], [northern], [−0.122], [−0.262], [+2.599],
    [], [central], [−0.078], [−0.156], [−1.056],
    [], [southern], [−0.222], [−0.301], [+2.780],
    [Puyseguer], [northern], [−0.056], [−0.156], [+1.101],
    [], [central], [−0.048], [−0.088], [−0.345],
    [], [southern], [−0.255], [−0.296], [*+7.459*],
    table.hline(),
  )
]

The two columns differ by #text(font: "DejaVu Sans Mono")[0.04]–#text(font: "DejaVu Sans Mono")[0.14] s
throughout — between 0.5% and 12% of the error they must be small against — so the cheap
constant-velocity control is a sound proxy for a correctly depth-sampled model, and the
equivalence is not an artefact of Hikurangi's shallow dip.

One qualification: the gap is *systematically signed*, the control always the less
negative, and as a fraction of the residual itself it understates by 14–64%. It is a
reliable way to establish that the refactored model's timing error is negligible, not a
precise estimate of what that residual is.

=== Geometry does not move; depth does

Splitting each hypocentre's error into the two terms shows where the *spatial* variability
lives, and it is not in the geometry:

- *Hikurangi*: the geometric term spans #text(font: "DejaVu Sans Mono")[0.015] s across
  the three hypocentres, the depth term #text(font: "DejaVu Sans Mono")[4.314] s — a
  factor of *285*.
- *Puysegur–Fiordland*: #text(font: "DejaVu Sans Mono")[0.144] against
  #text(font: "DejaVu Sans Mono")[3.980] s — a factor of 28.
- *Puyseguer*: #text(font: "DejaVu Sans Mono")[0.207] against
  #text(font: "DejaVu Sans Mono")[8.011] s — a factor of 39.

The geometric term does vary — it triples to quintuples at the southern hypocentre on both
Puysegur surfaces — but from a base so small that it never signifies. At the northern and
southern sites it carries the *opposite sign* to the total, so the whole of the positive
along-strike error is depth and geometry slightly offsets it. Its largest share anywhere
is 14%, at Puysegur central, where the total error is smallest.

This resolves what the Puysegur end-member numbers left ambiguous: their
#text(font: "DejaVu Sans Mono")[+1.10] and #text(font: "DejaVu Sans Mono")[+7.46] s errors
are depth, not geometry.

#figure(
  image("figures/decomposition_by_site.png"),
  caption: [Hikurangi, each hypocentre split into its geometric and depth terms.],
)

== Puysegur, where the picture inverts

The Puysegur interfaces carry roughly twice Hikurangi's curvature — `|grad h|` at the 90th
percentile is #text(font: "DejaVu Sans Mono")[0.77]–#text(font: "DejaVu Sans Mono")[0.88]
against #text(font: "DejaVu Sans Mono")[0.43]. On the reasoning so far they should be
worse. They are worse in one respect and *much better* in the other, and the reversal is
the most useful thing in this study.

#html.elem("div", attrs: (class: "numeric"))[
  #table(
    columns: (1fr, auto, auto, auto),
    stroke: none,
    align: (left, right, right, right),
    table.hline(),
    table.header([], [*Hikurangi*], [*Puyseguer*], [*Puysegur–Fiordland*]),
    table.hline(),
    [plane dip from vertical], [14.1°], [21.2°], [22.8°],
    [area ratio, mean], [1.0281], [*1.0655*], [*1.0580*],
    [area ratio, p90], [1.078], [1.194], [1.148],
    [rigidity contribution], [*0.9384*], [0.9861], [0.9821],
    [moment delivered / target], [0.9690], [*1.0680*], [*1.0342*],
    [onset error, median], [*+7.53 s*], [−0.345 s], [−1.056 s],
    [faces above sea level], [118,038], [241], [9,882],
    table.hline(),
  )
]

*On Hikurangi depth dominates; on Puysegur area dominates.* The area error more than
doubles, exactly as the extra curvature implies — but the depth error nearly vanishes, and
the rigidity contribution falls from #text(font: "DejaVu Sans Mono")[−6.2%] to
#text(font: "DejaVu Sans Mono")[−1.4%].

The reason is dip. The out-of-plane displacement $h$ becomes a *depth* error in proportion
to how nearly horizontal the reference plane is; Puysegur's plane stands 21–23° from
vertical against Hikurangi's 14°, so the same $h$ buys much less depth error. It is the
same $1 slash sin(d i p)$ amplification that makes steeply-dipping crustal faults immune to
this entire class of problem.

The consequence for the moment is that the two contributions stop cancelling. On Hikurangi
the area error and the rigidity error happened to oppose one another and left a 3.1%
residual; on Puyseguer they do not, and the flat model over-delivers by
#text(font: "DejaVu Sans Mono")[6.8%].

#figure(
  image("figures/puyseguer_onset_polar.png"),
  caption: [Puyseguer onset difference. Compare the Hikurangi polar plot: the quadrupole
    is far weaker because the steeper dip suppresses the depth term.],
)

#figure(
  image("figures/puyseguer_depth_error.png"),
  caption: [Puyseguer depth error. Larger curvature, but a steeper plane, so less of it
    becomes depth.],
)

#figure(
  image("figures/puyseguer_sections.png"),
  caption: [Puyseguer down-dip sections.],
)

`Puyseguer` as shipped is *refused* by the mesh-quality gate — it carries near-duplicate
vertices, the closest pair 0.2 m apart, which would leave the sampler's operator nearly
unconstrained there. Rebuilding the mesh repairs it, taking the worst lumped-mass ratio
from #text(font: "DejaVu Sans Mono")[7.3 × 10#super[−7]] to
#text(font: "DejaVu Sans Mono")[1/6], the ideal value. Both surfaces were run at Mw 8.5,
chosen from a ladder checked against Mai's fitted range rather than assumed from
Hikurangi.

=== What this means for the general case

Neither mechanism is universally dominant, and which one bites is set by the *dip* of the
fitted plane rather than by the curvature:

- a *shallow-dipping, large* interface — Hikurangi — turns $h$ into depth error, and depth
  dominates by two orders of magnitude;
- a *steeper, more curved* interface — Puysegur — keeps $h$ out of the depth term, and the
  area error dominates instead;
- a *steeply-dipping crustal fault* has neither, which is why this has never shown up in
  the shipped crustal examples.

A modeller cannot therefore reason from curvature alone about whether a planar
approximation is safe. Both terms need measuring on the surface in hand.

== What this adds up to

Setting the results against each other rather than reporting them one at a time:

- *The timing error is real, large, and fixable.* Up to 11.8 s of median error on
  Hikurangi, essentially all of it depth, and essentially all of it removed by reading
  material properties at the true depth. That argues for changing where material
  properties are assigned, not for abandoning the plane.
- *Half a fix is worse than none.* Correcting the velocity sampling while leaving the
  depth ramps on planar depths gives #text(font: "DejaVu Sans Mono")[+9.10] s, worse than
  the status quo.
- *The moment error is small and irreducible.* #text(font: "DejaVu Sans Mono")[3.3]–#text(font: "DejaVu Sans Mono")[6.8%],
  which sits inside magnitude and scaling uncertainty — and this is a single plane fitted
  to an *entire* interface, the least favourable geometry anyone would use. A scenario
  rupturing a portion gets a better-fitting plane and a smaller error.
- *One result is not a matter of degree.* The plane places
  #text(font: "DejaVu Sans Mono")[8.5%] of the Hikurangi interface — 14,943 km² — above
  sea level, reaching #text(font: "DejaVu Sans Mono")[17.6] km into the air. No error
  budget absorbs that, and no correction to material sampling touches it.

The case for modelling on the curved surface therefore rests on the geometry being
physically right and on not having to make the depth correction at all — not on the
few-percent moment difference this study began by measuring.

== Caveats

- *One realisation.* The four scenarios share seeds, so they are not independent samples
  and no uncertainty is attached to any figure here.
- *Seed-ball bias.* The eikonal solver's analytic seeding assumes constant slowness across
  the seeded ball; that costs up to 151 ms against a 50 ms budget at the northern site.
  It is present in both models and amounts to 0.4% of the effect measured.
- *Rise time floor.* At 0.02 s the rise-time field floors at exactly one sample interval,
  so 8.4% of faces sit at the floor carrying 0.8% of the moment. No moment is lost — the
  moment rate integral closes on the target to two parts in $10^9$ — but a finer sample
  interval would distribute it differently.
- *Rake and the onset perturbation read no depth at all*, contrary to what was expected
  when this study was specified. The only depth-dependent stages are rise time, rupture
  speed and the pulse's rising fraction. The rake difference between the models is purely
  the sampler's geometry.
- 525 m loses the 400–525 m octave of *slip* heterogeneity. It loses no geometry.

== Reproducing

#table(
  columns: (auto, 1fr),
  stroke: none,
  align: (left, left),
  table.hline(),
  [mesh], [1,389,600 faces, 525 m median edge, built at 2 km and subdivided twice],
  [magnitude], [Mw 8.5, Mai & Beroza correlation lengths (56.2 km strike, 21.5 km dip)],
  [sample interval], [0.02 s],
  [analysis], [324 s, 4.5 GB peak],
  [ruptures], [eight native files, 3.4 GB each, 399 M samples each],
  [source], [`curvature/`, with every quoted number in `results.json`],
  table.hline(),
)
