import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

const STORAGE_KEY = 'cs_user_id'

export const useUserStore = defineStore('user', () => {
  const userId = ref(loadUserId())

  // 从 localStorage 读取或生成新 ID
  function loadUserId(): string {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) return stored
    const generated = 'user_' + Date.now().toString(36)
    localStorage.setItem(STORAGE_KEY, generated)
    return generated
  }

  function setUserId(id: string) {
    userId.value = id
    localStorage.setItem(STORAGE_KEY, id)
  }

  return { userId, setUserId }
})
