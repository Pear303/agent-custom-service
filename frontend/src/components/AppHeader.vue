<template>
  <header class="header">
    <h1>AI 智能服务中心</h1>
    <div class="user-bar">
      用户ID:
      <input
        type="text"
        :value="userStore.userId"
        placeholder="输入你的ID"
        @change="handleChange"
      />
    </div>
  </header>
</template>

<script setup lang="ts">
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

function handleChange(e: Event) {
  const target = e.target as HTMLInputElement
  userStore.setUserId(target.value.trim() || userStore.userId)
}
</script>
