import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TicketSummary, TicketDetail, TicketStatus } from '@/types/api'
import { listTickets, getTicketStatus } from '@/api/task'

export const useTicketStore = defineStore('ticket', () => {
  const tickets = ref<TicketSummary[]>([])
  const currentTicket = ref<TicketDetail | null>(null)
  const loading = ref(false)
  let pollTimer: ReturnType<typeof setInterval> | null = null

  // 轮询拉取工单列表（diff 合并，避免整体替换导致闪烁）
  async function fetchTickets(userId: string) {
    try {
      const data = await listTickets(userId)
      const newList = data.tickets
      // 首次加载直接赋值
      if (tickets.value.length === 0) {
        tickets.value = newList
        return
      }
      // diff 合并：按 ticket_id 匹配，仅更新变化的项
      const map = new Map(tickets.value.map(t => [t.ticket_id, t]))
      let changed = false
      for (const newTicket of newList) {
        const old = map.get(newTicket.ticket_id)
        if (!old || JSON.stringify(old) !== JSON.stringify(newTicket)) {
          changed = true
          break
        }
      }
      // 新增或删除也算变化
      if (!changed && newList.length !== tickets.value.length) changed = true
      if (changed) tickets.value = newList
    } catch {
      // 静默失败
    }
  }

  async function fetchTicketDetail(ticketId: string, silent = false) {
    if (!silent) loading.value = true
    try {
      const data = await getTicketStatus(ticketId)
      if ('error' in data && data.error) {
        if (!silent) currentTicket.value = null
        return null
      }
      // 静默刷新时，仅当数据确实变化才更新，避免无意义重渲染
      if (silent && currentTicket.value) {
        if (JSON.stringify(currentTicket.value) === JSON.stringify(data)) return data as TicketDetail
      }
      currentTicket.value = data as TicketDetail
      return data as TicketDetail
    } catch {
      if (!silent) currentTicket.value = null
      return null
    } finally {
      if (!silent) loading.value = false
    }
  }

  function startPolling(userId: string, interval = 5000) {
    stopPolling()
    fetchTickets(userId)
    pollTimer = setInterval(() => fetchTickets(userId), interval)
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    tickets,
    currentTicket,
    loading,
    fetchTickets,
    fetchTicketDetail,
    startPolling,
    stopPolling,
  }
})
