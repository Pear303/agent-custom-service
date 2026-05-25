import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { TicketSummary, TicketDetail, TicketStatus } from '@/types/api'
import { listTickets, getTicketStatus } from '@/api/task'

export const useTicketStore = defineStore('ticket', () => {
  const tickets = ref<TicketSummary[]>([])
  const currentTicket = ref<TicketDetail | null>(null)
  const loading = ref(false)
  let pollTimer: ReturnType<typeof setInterval> | null = null

  // 轮询拉取工单列表
  async function fetchTickets(userId: string) {
    try {
      const data = await listTickets(userId)
      tickets.value = data.tickets
    } catch {
      // 静默失败
    }
  }

  async function fetchTicketDetail(ticketId: string) {
    loading.value = true
    try {
      const data = await getTicketStatus(ticketId)
      if ('error' in data && data.error) {
        currentTicket.value = null
        return null
      }
      currentTicket.value = data as TicketDetail
      return data as TicketDetail
    } catch {
      currentTicket.value = null
      return null
    } finally {
      loading.value = false
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
