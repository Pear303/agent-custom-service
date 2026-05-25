import type {ChatMessage} from '@/types/api'
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
})

export interface SessionHistoryResponse {
  user_id: string
  history: ChatMessage[]
  message_count: number
}

export async function getSessionHistory(userId: string): Promise<ChatMessage[]> {
  const { data } = await api.get<SessionHistoryResponse>('/session/history', {
    params: { user_id: userId },
  })
  return data.history || []
}