import axios from 'axios'
import type {
  RequirementRequest,
  TicketListResponse,
  TicketStatusResponse,
  SubmitResponse,
  ActionResponse,
  RestoreResponse,
} from '@/types/api'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
})

// 工单 CRUD API
export async function submitRequirement(req: RequirementRequest): Promise<SubmitResponse> {
  const { data } = await api.post<SubmitResponse>('/task/submit', req)
  return data
}

export async function listTickets(userId: string, limit = 50): Promise<TicketListResponse> {
  const { data } = await api.get<TicketListResponse>('/task/list', {
    params: { user_id: userId, limit },
  })
  return data
}

export async function getTicketStatus(ticketId: string): Promise<TicketStatusResponse> {
  const { data } = await api.get<TicketStatusResponse>(`/task/${ticketId}/status`)
  return data
}

export async function startDevelopment(ticketId: string): Promise<ActionResponse> {
  const { data } = await api.post<ActionResponse>(`/task/${ticketId}/start-development`)
  return data
}

export async function retryTicket(ticketId: string): Promise<ActionResponse> {
  const { data } = await api.post<ActionResponse>(`/task/${ticketId}/retry`)
  return data
}

export async function restoreLocalFiles(ticketId: string): Promise<RestoreResponse> {
  const { data } = await api.post<RestoreResponse>(`/task/${ticketId}/restore-local`)
  return data
}
