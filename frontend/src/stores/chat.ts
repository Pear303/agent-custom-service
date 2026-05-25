import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ChatMessage } from '@/types/api'
import { resetSession } from '@/api/health'
import { getSessionHistory } from '@/api/session'

const HISTORY_LIST_KEY_PREFIX = 'chat_history_list_'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const streaming = ref(false)
  const streamContent = ref('')
  const currentUserId = ref('')
  const chatHistory = ref<ChatMessage[][]>([])
  const chatHistoryLoaded = ref(false)

  /** 从后端加载当前会话消息，从 localStorage 加载历史会话列表 */
  async function loadHistory(userId: string) {
    currentUserId.value = userId
    if (chatHistoryLoaded.value) return
    try {
      const history = await getSessionHistory(userId)
      messages.value = history
    } catch {
      messages.value = []
    } finally {
      chatHistoryLoaded.value = true
    }
    loadChatHistoryFromLocal()
  }

  /* ── 历史会话列表（localStorage 持久化） ── */

  function loadChatHistoryFromLocal() {
    const key = HISTORY_LIST_KEY_PREFIX + currentUserId.value
    try {
      const stored = localStorage.getItem(key)
      if (stored) chatHistory.value = JSON.parse(stored)
    } catch {
      chatHistory.value = []
    }
  }

  function syncChatHistoryToLocal() {
    const key = HISTORY_LIST_KEY_PREFIX + currentUserId.value
    try {
      localStorage.setItem(key, JSON.stringify(chatHistory.value))
    } catch {
      // localStorage 空间不足
    }
  }

  /** 清空历史会话列表（不清当前消息） */
  function clearChatHistory() {
    chatHistory.value = []
    if (currentUserId.value) {
      localStorage.removeItem(HISTORY_LIST_KEY_PREFIX + currentUserId.value)
    }
  }

  /** 加载某条历史会话到当前消息框 */
  function setMessages(history: ChatMessage[]) {
    messages.value = history
  }

  /** 后端 AgentService 已自动保存 history，前端无需主动同步 */
  function saveHistory() {
  }

  /** 清空当前会话 → 保存到历史列表 → 重置后端 session */
  async function reset(userId: string) {
    if (messages.value.length > 0) {
      chatHistory.value.unshift([...messages.value])
      syncChatHistoryToLocal()
    }
    messages.value = []
    chatHistoryLoaded.value = false
    try {
      await resetSession(userId)
    } catch {
      // 尽力而为
    }
  }

  function setStreaming(val: boolean, content = '') {
    streaming.value = val
    streamContent.value = content
  }

  function updateStreamContent(content: string) {
    streamContent.value = content
  }

  function addMessage(msg: ChatMessage) {
    messages.value.push(msg)
  }

  const lastAssistantMessage = computed(() => {
    for (let i = messages.value.length - 1; i >= 0; i--) {
      if (messages.value[i].role === 'assistant') return messages.value[i]
    }
    return null
  })

  return {
    messages,
    streaming,
    streamContent,
    currentUserId,
    chatHistory,
    chatHistoryLoaded,
    lastAssistantMessage,
    loadHistory,
    setMessages,
    saveHistory,
    clearChatHistory,
    setStreaming,
    updateStreamContent,
    addMessage,
    reset,
  }
})


