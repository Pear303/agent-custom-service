<template>
  <div v-if="panels.length > 0">
    <div
      v-for="(panel, i) in panels"
      :key="i"
      class="local-panel"
      :style="panel.style"
    >
      <h3 :style="panel.titleStyle">{{ panel.title }}</h3>
      <div v-for="(row, j) in panel.rows" :key="j" class="local-row">
        <span class="dot" :class="row.dotClass"></span>
        <span v-html="row.text"></span>
        <button
          v-if="row.showRestore"
          class="restore-btn"
          @click="$emit('restore')"
        >
          恢复
        </button>
      </div>
      <div
        v-if="panel.fileListHtml"
        class="file-list"
        v-html="panel.fileListHtml"
      ></div>
      <button
        v-if="panel.showFullRestore"
        class="btn-primary"
        style="margin-top: 10px; width: 100%"
        @click="$emit('restore')"
      >
        🔄 从数据库恢复本地文件
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TicketDetail, LocalStatus } from '@/types/api'

const props = defineProps<{ ticket: TicketDetail }>()
defineEmits<{ restore: [] }>()

interface PanelDef {
  title: string
  style?: string
  titleStyle?: string
  rows: Array<{
    dotClass: string
    text: string
    showRestore?: boolean
  }>
  fileListHtml?: string
  showFullRestore?: boolean
}

// 根据 local_status 生成状态面板列表
const panels = computed<PanelDef[]>(() => {
  const ls: LocalStatus | null = props.ticket.local_status
  if (!ls) return []

  if (ls.local_deleted) {
    return [
      {
        title: '⚠️ 本地文件状态',
        style: 'border-color: #fee2e2; background: #fef2f2;',
        titleStyle: 'color: #991b1b;',
        rows: [
          {
            dotClass: 'red',
            text: '<strong>工单本地数据已被删除</strong>',
          },
          {
            dotClass: '',
            text: `数据库记录仍存在，但 data/users/${props.ticket.user_id}/${props.ticket.ticket_id}/ 目录不存在。`,
          },
        ],
        showFullRestore: true,
      },
    ]
  }

  if (ls.is_empty_workspace) {
    return [
      {
        title: '📁 本地文件状态',
        rows: [
          {
            dotClass: 'gray',
            text: '工单目录已创建，但尚未生成任何内容。',
          },
        ],
        showFullRestore: true,
      },
    ]
  }

  const result: PanelDef = {
    title: '📂 本地文件状态',
    rows: [],
  }

  result.rows.push({
    dotClass: ls.ticket_json_exists ? 'green' : 'red',
    text: `工单记录：${ls.ticket_json_exists ? '✅ 存在' : '❌ 缺失'}`,
  })

  if (ls.report_status === 'not_expected') {
    result.rows.push({
      dotClass: 'gray',
      text: '报告文件：⏳ 尚未生成',
    })
  } else {
    const expected = ls.expected_reports || []
    const actual = ls.report_files || []
    const missing = ls.missing_reports || []

    if (missing.length === 0) {
      result.rows.push({
        dotClass: 'green',
        text: `报告文件：✅ 完整 (${actual.length}/${expected.length})`,
      })
      result.fileListHtml = actual.map(f => '📄 ' + f).join('<br>')
    } else {
      result.rows.push({
        dotClass: 'yellow',
        text: `报告文件：⚠️ 部分缺失 (${actual.length}/${expected.length})`,
        showRestore: true,
      })
      const lines: string[] = []
      for (const f of actual) lines.push('<span style="color:#10b981;">✅ ' + f + '</span>')
      for (const f of missing) lines.push('<span style="color:#ef4444;">❌ ' + f + '</span>')
      result.fileListHtml = lines.join('<br>')
    }
  }

  if (ls.has_product) {
    result.rows.push({
      dotClass: 'green',
      text: `成品文件：✅ ${ls.product_file_count} 个`,
    })
    if (ls.product_sample && ls.product_sample.length > 0) {
      const sampleHtml = ls.product_sample.map(f => '📦 ' + f).join('<br>')
      const suffix = ls.product_file_count > 5
        ? '<br><span style="color: #94a3b8;">... 还有 ' + (ls.product_file_count - 5) + ' 个文件</span>'
        : ''
      result.fileListHtml = (result.fileListHtml || '') + sampleHtml + suffix
    }
  } else if (props.ticket.status === 'development_completed') {
    result.rows.push({
      dotClass: 'red',
      text: '成品文件：❌ 缺失（开发已完成但本地文件被删除）',
      showRestore: true,
    })
  } else if (props.ticket.status === 'developing') {
    result.rows.push({
      dotClass: 'gray',
      text: '成品文件：⏳ 开发进行中...',
    })
  } else {
    result.rows.push({
      dotClass: 'gray',
      text: '成品文件：⏳ 尚未生成',
    })
  }

  return [result]
})
</script>
