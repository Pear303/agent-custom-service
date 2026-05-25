<template>
  <div class="chat-input">
    <input
      ref="inputRef"
      type="text"
      v-model="text"
      placeholder="输入你的问题..."
      autocomplete="off"
      :disabled="disabled"
      @keydown="handleKeydown"
    />
    <button :disabled="disabled || !text.trim()" @click="emitSend">发送</button>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [msg: string] }>()

const text = ref('')
const inputRef = ref<HTMLInputElement>()

// Enter 发送，Shift+Enter 换行
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && (!e.shiftKey || !e.ctrlKey)) {
    e.preventDefault()
    emitSend()
  }
}

function emitSend() {
  const msg = text.value.trim()
  if (!msg) return
  emit('send', msg)
  text.value = ''
  nextTick(() => inputRef.value?.focus())
}
</script>
