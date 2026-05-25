import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './assets/main.css'

// 挂载 Pinia 状态管理、路由、启动应用
const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
