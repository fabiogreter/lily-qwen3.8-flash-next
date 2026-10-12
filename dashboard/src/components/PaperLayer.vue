<script setup lang="ts">
// One layer behind the whole page, drawn once at the page's size and scrolling
// with it, so it never tiles or seams. The lamp glow always; the texture by
// the `data-paper` attribute on <html>: smooth (none), grain (fine even
// specks), charcoal (soft mottle plus grain).
</script>

<template>
  <div class="paper-layer" aria-hidden="true">
    <svg preserveAspectRatio="none">
      <defs>
        <!-- Fine isotropic grain: noise thresholded into sparse specks. The noise is only the mask; the rect's fill (SourceGraphic) is the colour, so dark mode's light ink shows on charcoal. -->
        <filter id="paper-grain" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" seed="3" result="n" />
          <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  2.6 0 0 0 -1.25" result="mask" />
          <feComposite in="SourceGraphic" in2="mask" operator="in" />
        </filter>
        <!-- Soft mottle: low-frequency clouds, like the uneven tone of charcoal paper. -->
        <filter id="paper-mottle" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.0035 0.0045" numOctaves="4" seed="21" result="n" />
          <feColorMatrix in="n" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  1.8 0 0 0 -0.75" result="mask" />
          <feComposite in="SourceGraphic" in2="mask" operator="in" />
        </filter>
      </defs>
      <rect class="mottle" width="100%" height="100%" filter="url(#paper-mottle)" />
      <rect class="grain" width="100%" height="100%" filter="url(#paper-grain)" />
    </svg>
  </div>
</template>

<style>
.paper-layer {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background: radial-gradient(1400px 900px at 18% -10%, var(--paper-lamp), transparent 70%);
}
.paper-layer svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.paper-layer .grain {
  fill: var(--grain-ink);
  opacity: var(--grain-opacity);
}
.paper-layer .mottle {
  fill: var(--mottle-ink);
  opacity: var(--mottle-opacity);
}
html[data-paper='smooth'] .paper-layer svg {
  display: none;
}
html[data-paper='grain'] .paper-layer .mottle {
  display: none;
}
</style>
