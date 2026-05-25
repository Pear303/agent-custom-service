<template>
  <div class="panel active">
    <div class="form-card">
      <button class="btn-secondary" @click="$router.push({ name: 'requirement' })" style="margin-bottom: 16px">
        ← 返回工单列表
      </button>

      <div v-if="ticketStore.loading" style="text-align: center; padding: 40px; color: var(--color-text-muted)">
        加载中...
      </div>

      <template v-else-if="ticket">
        <h2>{{ ticket.project_name || '未命名项目' }}</h2>
        <p class="subtitle">工单号: {{ ticket.ticket_id }} · 状态: {{ statusText }}</p>

        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: progressPct + '%' }"></div>
        </div>
        <p style="text-align: center; font-size: 13px; color: #64748b">
          处理进度: {{ progressPct }}%
        </p>

        <ResultSection
          v-if="ticket.analysis"
          title="📋 需求分析"
          :content="ticket.analysis"
        />
        <p v-else style="color: #64748b; margin-top: 16px">需求分析尚未完成...</p>

        <ResultSection
          v-if="ticket.prd"
          title="📝 产品需求文档 (PRD)"
          :content="ticket.prd"
        />

        <ResultSection
          v-if="ticket.quote"
          title="💰 成本估算"
          :content="ticket.quote"
        />

        <div class="form-actions">
          <button
            v-if="ticket.status === 'pending_development'"
            class="btn-primary"
            style="width: 100%"
            @click="handleStartDev"
          >
            🚀 开始开发
          </button>
          <button
            v-if="ticket.status === 'development_failed'"
            class="btn-primary"
            style="width: 100%"
            @click="handleStartDev"
          >
            🔄 重新开发
          </button>
          <button
            v-if="ticket.status === 'failed'"
            class="btn-primary"
            style="width: 100%"
            @click="handleRetry"
          >
            🔄 重新处理
          </button>
        </div>

        <div
          v-if="ticket.status === 'developing'"
          style="background: #dbeafe; border: 1px solid #3b82f6; border-radius: 8px; padding: 12px; margin-top: 16px; color: #1e40af"
        >
          ⚙️ 开发进行中，请稍候...
        </div>

        <template v-if="ticket.development_output">
          <ResultSection title="💻 开发结果">
            <div v-if="ticket.development_output.project_structure">
              <h4>项目结构</h4>
              <pre>{{ ticket.development_output.project_structure }}</pre>
            </div>
            <div v-if="ticket.development_output.tech_stack">
              <h4>技术栈</h4>
              <pre>{{ JSON.stringify(ticket.development_output.tech_stack, null, 2) }}</pre>
            </div>
            <div v-if="ticket.development_output.setup_instructions">
              <h4>安装与运行</h4>
              <pre>{{ ticket.development_output.setup_instructions }}</pre>
            </div>
            <div v-if="ticket.development_output.files">
              <h4>生成的文件 ({{ ticket.development_output.files.length }} 个)</h4>
              <pre>{{ JSON.stringify(ticket.development_output.files.map(f => f.path), null, 2) }}</pre>
            </div>
          </ResultSection>
        </template>

        <ResultSection
          v-if="ticket.development_error"
          title="❌ 开发错误"
          :content="ticket.development_error"
          :error="true"
        />

        <ResultSection
          v-if="ticket.error && !ticket.development_error"
          title="❌ 错误信息"
          :content="ticket.error"
          :error="true"
        />

        <LocalStatusPanel :ticket="ticket" @restore="handleRestoreLocal" />

        <p style="font-size: 12px; color: #94a3b8; margin-top: 16px">
          创建时间: {{ ticket.created_at || '未知' }} · 更新时间: {{ ticket.updated_at || '未知' }}
        </p>
      </template>

      <div v-else style="text-align: center; padding: 40px; color: var(--color-error)">
        工单不存在或加载失败
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useTicketStore } from '@/stores/ticket'
import { useToastStore } from '@/stores/toast'
import { startDevelopment, retryTicket, restoreLocalFiles } from '@/api/task'
import ResultSection from '@/components/ResultSection.vue'
import LocalStatusPanel from '@/components/LocalStatusPanel.vue'

const props = defineProps<{ id: string }>()
const router = useRouter()
const ticketStore = useTicketStore()
const toastStore = useToastStore()

const ticket = computed(() => ticketStore.currentTicket)

// 工单状态中文映射（与 TicketCard 一致）
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

const statusText = computed(() => {
  if (!ticket.value) return ''
  return STATUS_MAP[ticket.value.status] || ticket.value.status
})

const progressPct = computed(() => {
  if (!ticket.value) return 0
  return Math.max(0, ticket.value.progress)
})

let pollTimer: ReturnType<typeof setInterval> | null = null

async function loadDetail() {
  await ticketStore.fetchTicketDetail(props.id)
}

async function handleStartDev() {
  try {
    const result = await startDevelopment(props.id)
    if (result.status === 'developing') {
      toastStore.show('开发已启动，正在生成代码...')
      await loadDetail()
    } else {
      toastStore.show(result.error || '启动开发失败', 'error')
    }
  } catch {
    toastStore.show('网络错误，请稍后重试', 'error')
  }
}

async function handleRetry() {
  try {
    const result = await retryTicket(props.id)
    if (result.status === 'queued') {
      toastStore.show('工单已重新提交，正在排队处理...')
      await loadDetail()
    } else {
      toastStore.show(result.error || '重试失败', 'error')
    }
  } catch {
    toastStore.show('网络错误，请稍后重试', 'error')
  }
}

async function handleRestoreLocal() {
  try {
    toastStore.show('正在恢复本地文件...')
    const result = await restoreLocalFiles(props.id)
    if (result.status === 'ok') {
      const r = result.restored
      const parts: string[] = []
      if (r.ticket_json) parts.push('工单')
      if (r.reports.length > 0) parts.push(`报告(${r.reports.length})`)
      if (r.products > 0) parts.push(`成品(${r.products})`)
      toastStore.show('恢复完成: ' + (parts.length > 0 ? parts.join(', ') : '无内容可恢复'))
      await loadDetail()
    } else {
      toastStore.show('恢复失败: ' + (result.error || '未知错误'), 'error')
    }
  } catch {
    toastStore.show('网络错误，请稍后重试', 'error')
  }
}

// 进入页面加载详情，每 5 秒轮询更新
onMounted(() => {
  loadDetail()
  pollTimer = setInterval(loadDetail, 5000)
})

// 离开页面停止轮询
onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>
