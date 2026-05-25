<template>
  <div class="panel active">
    <div class="form-card">
      <h2>需求对接窗口</h2>
      <p class="subtitle">描述你的项目需求，我们的 AI 团队会为你分析并给出初步方案。</p>

      <div class="form-group">
        <label>项目名称 <span class="req">*</span></label>
        <input v-model="form.project_name" type="text" placeholder="例如：电商小程序" />
      </div>

      <div class="form-group">
        <label>项目类型</label>
        <select v-model="form.project_type">
          <option value="">请选择</option>
          <option value="web">网站开发</option>
          <option value="miniapp">小程序</option>
          <option value="app">移动 App</option>
          <option value="automation">自动化脚本</option>
          <option value="data">数据分析</option>
          <option value="ai">AI 应用</option>
          <option value="other">其他</option>
        </select>
      </div>

      <div class="form-group">
        <label>需求描述 <span class="req">*</span></label>
        <textarea
          v-model="form.description"
          placeholder="请详细描述你的需求，包括目标用户、核心功能、预算范围等..."
        ></textarea>
      </div>

      <div class="form-group">
        <label>期望交付时间</label>
        <select v-model="form.deadline">
          <option value="">请选择</option>
          <option value="1week">1周内</option>
          <option value="2weeks">2周内</option>
          <option value="1month">1个月内</option>
          <option value="3months">3个月内</option>
          <option value="flexible">灵活</option>
        </select>
      </div>

      <div class="form-group">
        <label>预算范围</label>
        <select v-model="form.budget">
          <option value="">请选择</option>
          <option value="5k">&lt; 5,000 元</option>
          <option value="5k-20k">5,000 - 20,000 元</option>
          <option value="20k-50k">20,000 - 50,000 元</option>
          <option value="50k+">&gt; 50,000 元</option>
          <option value="discuss">待商议</option>
        </select>
      </div>

      <div class="form-actions">
        <button class="btn-primary" @click="handleSubmit">提交需求</button>
        <button class="btn-secondary" @click="clearForm">清空</button>
      </div>
    </div>

    <div class="form-card" style="margin-top: 20px">
      <h2>我的工单</h2>
      <p class="subtitle">查看已提交的需求处理进度和结果。</p>
      <div v-if="ticketStore.tickets.length === 0" style="color: #64748b; text-align: center; padding: 20px">
        暂无工单，请先提交需求。
      </div>
      <TicketCard
        v-for="ticket in ticketStore.tickets"
        :key="ticket.ticket_id"
        :ticket="ticket"
        @click="goDetail(ticket.ticket_id)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '@/stores/user'
import { useTicketStore } from '@/stores/ticket'
import { useToastStore } from '@/stores/toast'
import { submitRequirement } from '@/api/task'
import TicketCard from '@/components/TicketCard.vue'

const router = useRouter()
const userStore = useUserStore()
const ticketStore = useTicketStore()
const toastStore = useToastStore()

const emptyForm = () => ({
  user_id: userStore.userId,
  project_name: '',
  project_type: '',
  description: '',
  deadline: '',
  budget: '',
})

const form = reactive(emptyForm())

// 提交需求表单
async function handleSubmit() {
  if (!form.project_name.trim() || !form.description.trim()) {
    toastStore.show('请填写项目名称和需求描述', 'error')
    return
  }
  try {
    const data = await submitRequirement({ ...form, user_id: userStore.userId })
    if (data.ticket_id) {
      toastStore.show(`需求已提交！工单号: ${data.ticket_id}`)
      clearForm()
      await ticketStore.fetchTickets(userStore.userId)
    } else {
      toastStore.show(`提交失败: ${(data as any).error || '未知错误'}`, 'error')
    }
  } catch {
    toastStore.show('网络错误，请稍后重试', 'error')
  }
}

function clearForm() {
  Object.assign(form, emptyForm())
}

function goDetail(ticketId: string) {
  router.push(`/ticket/${ticketId}`)
}

// 进入页面开始轮询，离开时停止
onMounted(() => {
  ticketStore.startPolling(userStore.userId)
})

onUnmounted(() => {
  ticketStore.stopPolling()
})
</script>
