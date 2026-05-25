<template>
  <div class="msg" :class="message.role">
    <span v-html="contentHtml"></span>
    <span v-if="sourceLabel" class="source">{{ sourceLabel }}</span>
    <span v-if="timestampLabel" class="timestamp">{{ timestampLabel }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '@/types/api'

const props = defineProps<{ message: ChatMessage }>()

const contentHtml = computed(() => {
  return escapeHtml(props.message.content).replace(/\n/g, '<br>')
})

const sourceLabel = computed(() => {
  if (!props.message.source) return ''
  if (props.message.source === 'dify') return 'AI 客服'
  if (props.message.source === '系统错误') return '系统错误'
  return props.message.source === 'agent' ? '系统兜底 测试中' : props.message.source
})

const timestampLabel = computed(() => {
  console.log('timestampLabel 执行了, timestamp=', props.message.timestamp, '类型=', typeof props.message.timestamp)
  if (!props.message.timestamp) {
    console.log('timestamp 是 falsy 值，不显示')
    return ''
  }

  const date = new Date(props.message.timestamp)
  const now = new Date()
  const isToday = date.toDateString() === now.toDateString()

  if (isToday) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  else{
    return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
  }
})

// 防 XSS：转义 HTML 特殊字符
function escapeHtml(text: string): string {
  const d = document.createElement('div')
  d.textContent = text
  return d.innerHTML
}
</script>
