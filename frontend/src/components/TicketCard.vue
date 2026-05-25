<template>
  <div
    class="ticket-item"
    @click="$emit('click')"
  >
    <div class="ticket-header">
      <span class="title">{{ ticket.project_name || '未命名项目' }}</span>
      <span class="status" :class="'status-' + ticket.status">{{ statusText }}</span>
    </div>
    <div class="meta">{{ ticket.ticket_id }} · {{ ticket.created_at || '未知时间' }}</div>
    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: progressPct + '%' }"></div>
    </div>
    <div class="local-tags">
      <LocalStatusTags :ticket="ticket" @restore="$emit('restore', ticket.ticket_id)" />
    </div>
    <div v-if="ticket.status === 'pending_development'" style="font-size: 12px; color: #92400e; margin-top: 4px">
      ⚠️ 需求分析完成，请查看详情并确认是否开始开发
    </div>
    <div v-if="ticket.status === 'failed'" style="font-size: 12px; color: #991b1b; margin-top: 4px">
      ⚠️ 处理失败，点击查看详情可重新处理
    </div>
    <div v-if="ticket.error && ticket.status !== 'failed'" style="font-size: 12px; color: #991b1b; margin-top: 4px">
      错误：{{ ticket.error }}
    </div>
    <div v-if="ticket.development_error" style="font-size: 12px; color: #991b1b; margin-top: 4px">
      开发错误：{{ ticket.development_error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TicketSummary } from '@/types/api'
import LocalStatusTags from './LocalStatusTags.vue'

const props = defineProps<{ ticket: TicketSummary }>()
defineEmits<{ click: []; restore: [ticketId: string] }>()

// 工单状态中文映射
const STATUS_MAP: Record<string, string> = {
  queued: '排队中',
  analyzing: '需求分析中',
  designing: 'PRD 设计中',
  estimating: '成本估算中',
  pending_development: '待确认开发',
  developing: '开发中',
  development_completed: '开发完成',
  development_failed: '开发失败',
  completed: '已完成',
  failed: '处理失败',
}

const statusText = computed(() => STATUS_MAP[props.ticket.status] || props.ticket.status)
const progressPct = computed(() => Math.max(0, props.ticket.progress || 0))
</script>
