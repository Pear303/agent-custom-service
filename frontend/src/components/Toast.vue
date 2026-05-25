<template>
  <Teleport to="body">
    <div
      class="toast"
      :class="[toastStore.type, { show: toastStore.visible }]"
    >
      {{ toastStore.message }}
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { useToastStore } from '@/stores/toast'

const toastStore = useToastStore()
</script>
