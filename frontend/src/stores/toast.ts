import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ToastType = 'success' | 'error'

export const useToastStore = defineStore('toast', () => {
  const visible = ref(false)
  const message = ref('')
  const type = ref<ToastType>('success')
  let timer: ReturnType<typeof setTimeout> | null = null

  // 显示提示，3 秒后自动消失
  function show(msg: string, toastType: ToastType = 'success') {
    if (timer) clearTimeout(timer)
    message.value = msg
    type.value = toastType
    visible.value = true
    timer = setTimeout(() => {
      visible.value = false
    }, 3000)
  }

  return { visible, message, type, show }
})
