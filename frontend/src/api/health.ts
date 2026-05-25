import axios from 'axios'
import type { SessionResetResponse, HealthResponse } from '@/types/api'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
})

// 重置会话 & 健康检查
export async function resetSession(userId: string): Promise<SessionResetResponse> {
  const { data } = await api.post<SessionResetResponse>('/session/reset', null, {
    params: { user_id: userId },
  })
  return data
}

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health', { timeout: 5000 })
  return data
}
