import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: () => import('@/views/ChatView.vue'),
    },
    {
      path: '/requirement',
      name: 'requirement',
      component: () => import('@/views/RequirementView.vue'),
    },
    {
      path: '/ticket/:id',
      name: 'ticket-detail',
      component: () => import('@/views/TicketDetailView.vue'),
      props: true,
    },
  ],
})

export default router
