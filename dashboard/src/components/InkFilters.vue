<template>
  <!-- The shared ink filters, referenced by id from the brush shapes. -->
  <svg width="0" height="0" style="position: absolute" aria-hidden="true" focusable="false">
    <defs>
      <!-- Strong brush: ensō, seal edge. -->
      <filter id="ink-strong" filterUnits="userSpaceOnUse" x="0" y="0" width="320" height="320">
        <feTurbulence type="fractalNoise" baseFrequency="0.045" numOctaves="3" seed="4" result="n" />
        <feDisplacementMap in="SourceGraphic" in2="n" scale="5" xChannelSelector="R" yChannelSelector="G" result="d" />
        <feTurbulence type="fractalNoise" baseFrequency="0.6" numOctaves="2" seed="9" result="grain" />
        <feColorMatrix in="grain" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -0.7 1.38" result="dry" />
        <feComposite in="d" in2="dry" operator="in" />
      </filter>
      <!-- Divider brush: displacement only, solid ink. -->
      <filter id="ink-stroke" x="-2%" y="-60%" width="104%" height="220%">
        <feTurbulence type="fractalNoise" baseFrequency="0.015 0.08" numOctaves="2" seed="8" result="n" />
        <feDisplacementMap in="SourceGraphic" in2="n" scale="2" xChannelSelector="R" yChannelSelector="G" />
      </filter>
      <!-- Mild brush for data lines: keeps them readable. -->
      <filter id="ink-line" x="-2%" y="-20%" width="104%" height="140%">
        <feTurbulence type="fractalNoise" baseFrequency="0.08" numOctaves="2" seed="2" result="n" />
        <feDisplacementMap in="SourceGraphic" in2="n" scale="1.4" xChannelSelector="R" yChannelSelector="G" />
      </filter>
      <filter id="seal-rough" x="-10%" y="-10%" width="120%" height="120%">
        <feTurbulence type="fractalNoise" baseFrequency="0.12" numOctaves="2" seed="7" result="n" />
        <feDisplacementMap in="SourceGraphic" in2="n" scale="2.6" xChannelSelector="R" yChannelSelector="G" result="d" />
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="5" result="g" />
        <feColorMatrix in="g" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -2.2 1.9" result="speck" />
        <feComposite in="d" in2="speck" operator="in" />
      </filter>
    </defs>
  </svg>
</template>
