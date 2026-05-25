<template>
  <div class="panel active chat-layout">
    <!-- 聊天对话框 --><
    <div class="chat-container">
      <div class="chat-header">
        <span class="chat-header-title">AI 智能客服</span>
        <button class="chat-reset-btn" @click="handleReset">清空当前会话</button>
      </div>
      <div ref="messagesRef" class="chat-messages">
        <ChatMessage
          v-for="(msg, index) in chatStore.messages"
          :key="index"
          :message="msg"
        />
        <div v-if="chatStore.streaming" class="msg assistant" ref="streamRef">
          <div v-if="!chatStore.streamContent" class="typing">
            <span></span><span></span><span></span>
          </div>
          <span v-else v-html="streamHtml"></span>
        </div>
      </div>
      <ChatInput :disabled="chatStore.streaming" @send="handleSend" />
    </div>

    <!-- 会话记录 -->
     <div class="chat-history">
      <div class="chat-history-header">
        <span class="chat-history-header-title">历史记录</span>
        <button class="chat-history-reset-btn" @click="handleChatHistoryReset">清空全部记录</button>
      </div>
      <div class="chat-history-list">
        <div
          v-for="(item, index) in chatStore.chatHistory"
          :key="index"
          class="chat-history-item"
          @click="chatStore.setMessages(item)"
        >
          <div class="chat-history-item-title">{{ item[0].content }}</div>
          <div class="chat-history-item-time">{{ formatTime(item[item.length - 1].timestamp) }}</div>
        </div> 
      </div>
     </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref, nextTick, onMounted } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useUserStore } from '@/stores/user'
import { chatStream, chatBlocking } from '@/api/chat'
import ChatMessage from '@/components/ChatMessage.vue'
import ChatInput from '@/components/ChatInput.vue'

const chatStore = useChatStore()
const userStore = useUserStore()
const messagesRef = ref<HTMLElement>()
const streamRef = ref<HTMLElement>()
let pollTimer: ReturnType<typeof setInterval> | null = null

const streamHtml = computed(() => {
  return chatStore.streamContent.replace(/\n/g, '<br>')
})

// 发送消息：优先 SSE 流式，失败则回退非流式
async function handleSend(msg: string) {
  chatStore.addMessage({ role: 'user', content: msg, timestamp: Date.now() })
  chatStore.setStreaming(true, '')

  try {
    const result = await chatStream(userStore.userId, msg, (content, source) => {
      chatStore.updateStreamContent(content)
      nextTick(() => scrollToBottom())
    })
    chatStore.setStreaming(false)
    chatStore.addMessage({
      role: 'assistant',
      content: result.answer,
      source: result.source,
      timestamp: Date.now(),
    })
  } catch {
    try {
      const result = await chatBlocking(userStore.userId, msg)
      chatStore.setStreaming(false)
      chatStore.addMessage({
        role: 'assistant',
        content: result.answer,
        source: result.source,
        timestamp: Date.now(),
      })
    } catch {
      chatStore.setStreaming(false)
      chatStore.addMessage({
        role: 'assistant',
        content: '抱歉，服务暂时不可用，请稍后重试。',
        source: '系统错误',
        timestamp: Date.now(),
      })
    }
  }
  nextTick(() => scrollToBottom())
}

// 清空当前会话（保存到历史列表后再重置后端）
function handleReset() {
  if (confirm('确定要清空当前会话吗？（会话将保存到历史记录）')) {
    chatStore.reset(userStore.userId)
  }
}

// 清空全部历史记录（不影响当前会话）
function handleChatHistoryReset() {
  if (confirm('确定要清空全部历史记录吗？（不可恢复）')) {
    if (confirm('再次确认：清空所有历史记录？')) {
      chatStore.clearChatHistory()
    }
  }
}

// 格式化时间戳为短日期+时间
function formatTime(ts: number): string {
  const d = new Date(ts)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`
}

// 滚动到对话底部
function scrollToBottom() {
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  }
}

onMounted(() => {
  chatStore.loadHistory(userStore.userId)
  scrollToBottom()
})
</script>
