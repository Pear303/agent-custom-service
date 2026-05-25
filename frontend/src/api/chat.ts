import axios from 'axios'
import type { ChatRequest, ChatResponse } from '@/types/api'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
})

export async function chatBlocking(userId: string, message: string): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', {
    user_id: userId,
    message,
    stream: false,
  } as ChatRequest)
  return data
}

export async function chatStream(
  userId: string,
  message: string,
  onChunk: (content: string, source: string) => void,
): Promise<ChatResponse> {
  const resp = await fetch(
    (import.meta.env.VITE_API_BASE || '') + '/chat/stream',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, message } as ChatRequest),
    },
  )
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let fullAnswer = ''
  let source = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split(/\n\n|\n(?=data: )/)
    buffer = parts.pop()!

    for (const part of parts) {
      const lines = part.split('\n')
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6))
          if (data.event === 'message') {
            fullAnswer += data.answer || ''
            source = data.source || source
            onChunk(fullAnswer, source)
          }
        } catch {
          // 跳过格式异常的块
        }
      }
    }
  }

  return {
    user_id: userId,
    answer: fullAnswer,
    conversation_id: null,
    source: source as 'dify' | 'agent',
  }
}
