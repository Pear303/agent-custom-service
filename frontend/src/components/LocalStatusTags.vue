<template>
  <template v-if="ls">
    <span v-if="ls.local_deleted" class="local-tag deleted">
      ⚠ 本地已删除
      <button class="restore-btn" @click.stop="$emit('restore')">恢复</button>
    </span>
    <span v-else-if="ls.is_empty_workspace" class="local-tag empty">
      📁 空目录
      <button class="restore-btn" @click.stop="$emit('restore')">恢复</button>
    </span>
    <template v-else>
      <span v-if="ls.ticket_json_exists" class="local-tag ticket">📋 工单</span>
      <span v-if="ls.report_status === 'complete'" class="local-tag report">
        📄 报告({{ ls.report_files?.length || 0 }})
      </span>
      <span v-else-if="ls.report_status === 'partial'" class="local-tag missing-report">
        📄 报告({{ ls.report_files?.length || 0 }}/{{ ls.expected_reports?.length || 0 }}) ⚠️
        <button class="restore-btn" @click.stop="$emit('restore')">恢复</button>
      </span>
      <span v-else-if="ls.report_status === 'missing'" class="local-tag missing-report">
        📄 报告缺失
        <button class="restore-btn" @click.stop="$emit('restore')">恢复</button>
      </span>
      <span v-if="ls.has_product" class="local-tag product">📦 成品({{ ls.product_file_count }})</span>
      <span
        v-else-if="ticket.status === 'development_completed'"
        class="local-tag missing-product"
      >
        📦 成品缺失
        <button class="restore-btn" @click.stop="$emit('restore')">恢复</button>
      </span>
    </template>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TicketSummary } from '@/types/api'

const props = defineProps<{ ticket: TicketSummary }>()
defineEmits<{ restore: [] }>()

// 本地文件状态标签（删除/缺失/完整）
const ls = computed(() => props.ticket.local_status)
</script>
