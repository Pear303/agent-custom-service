<template>
  <div
    class="result-section"
    :style="error ? 'border-color: #fee2e2; background: #fef2f2;' : ''"
  >
    <h3 :style="error ? 'color: #991b1b;' : ''">{{ title }}</h3>
    <pre v-if="typeof content === 'string'">{{ content }}</pre>
    <pre v-else>{{ JSON.stringify(content, null, 2) }}</pre>
    <slot />
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  content?: unknown
  error?: boolean
}>()
</script>
