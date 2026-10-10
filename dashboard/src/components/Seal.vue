<script setup lang="ts">
// Hanko-style indigo square with a lily mark.
withDefaults(defineProps<{ size?: number }>(), { size: 48 });
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 100 100" role="img" aria-label="lily seal">
    <g filter="url(#seal-rough)">
      <rect x="4" y="4" width="92" height="92" rx="9" fill="var(--ai-seal)" />
      <rect x="12" y="12" width="76" height="76" rx="5" fill="none" stroke="#f3eee2" stroke-width="2.4" />
      <path d="M50 20 C59 31 59 48 50 60 C41 48 41 31 50 20Z" fill="#f3eee2" />
      <path d="M47 58 C35 55 25 45 21 31 C33 35 44 45 47 58Z" fill="#f3eee2" />
      <path d="M53 58 C65 55 75 45 79 31 C67 35 56 45 53 58Z" fill="#f3eee2" />
      <path d="M50 60 C50 69 48 76 44 84" fill="none" stroke="#f3eee2" stroke-width="3.2" stroke-linecap="round" />
      <path d="M47 76 C39 75 33 70 30 64 C38 65 44 70 47 76Z" fill="#f3eee2" />
    </g>
  </svg>
</template>
