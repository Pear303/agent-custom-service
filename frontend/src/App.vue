<template>
  <div>
    <AppHeader />
    <nav class="tabs">
      <div
        v-for="tab in tabs"
        :key="tab.name"
        class="tab"
        :class="{ active: currentRoute === tab.route }"
        @click="navigate(tab.route)"
      >
        {{ tab.label }}
      </div>
    </nav>
    <main>
      <router-view />
    </main>
    <Toast />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import AppHeader from '@/components/AppHeader.vue'
import Toast from '@/components/Toast.vue'

const router = useRouter()
const route = useRoute()

// 顶部标签导航项
const tabs = [
  { label: '智能客服', route: '/', name: 'chat' },
  { label: '需求对接', route: '/requirement', name: 'requirement' },
]

const currentRoute = computed(() => {
  if (route.path.startsWith('/ticket')) return '/requirement'
  return route.path
})

function navigate(path: string) {
  router.push(path)
}
</script>
