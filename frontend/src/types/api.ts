export interface ChatRequest {
  user_id: string
  message: string
  stream?: boolean
}

export interface ChatResponse {
  user_id: string
  answer: string
  conversation_id: string | null
  source: 'dify' | 'agent'
}

export interface SSEChunk {
  event: 'message' | 'message_end'
  answer?: string
  source?: string
}

export interface RequirementRequest {
  user_id: string
  project_name: string
  project_type: string
  description: string
  deadline: string
  budget: string
}

export interface TicketSummary {
  ticket_id: string
  user_id: string
  project_name: string
  status: TicketStatus
  progress: number
  created_at: string
  updated_at: string
  local_status: LocalStatus | null
  error?: string
  development_error?: string
}

export interface TicketDetail extends TicketSummary {
  project_type: string
  description: string
  deadline: string
  budget: string
  analysis: Record<string, unknown> | null
  prd: Record<string, unknown> | null
  quote: Record<string, unknown> | null
  development_output: DevelopmentOutput | null
  development_error: string | null
  error: string | null
}

export interface DevelopmentOutput {
  project_structure?: string
  tech_stack?: Record<string, unknown>
  setup_instructions?: string
  files?: Array<{
    path: string
    content?: string
    code?: string
    file?: string
    name?: string
  }>
}

export interface LocalStatus {
  local_deleted?: boolean
  is_empty_workspace?: boolean
  ticket_json_exists?: boolean
  report_status?: 'complete' | 'partial' | 'missing' | 'not_expected'
  report_files?: string[]
  expected_reports?: string[]
  missing_reports?: string[]
  has_product?: boolean
  product_file_count?: number
  product_sample?: string[]
}

export type TicketStatus =
  | 'queued'
  | 'analyzing'
  | 'designing'
  | 'estimating'
  | 'pending_development'
  | 'developing'
  | 'development_completed'
  | 'development_failed'
  | 'completed'
  | 'failed'

export interface TicketListResponse {
  tickets: TicketSummary[]
}

export interface TicketStatusResponse extends TicketDetail {}

export interface SubmitResponse {
  ticket_id: string
  status: string
  message: string
}

export interface ActionResponse {
  status: string
  message?: string
  error?: string
}

export interface RestoreResponse {
  status: string
  restored: {
    ticket_json: boolean
    reports: string[]
    products: number
  }
  local_status: LocalStatus
  error?: string
}

export interface SessionResetResponse {
  user_id: string
  status: string
}

export interface ServiceStatus {
  active_sessions: number
  dify_status: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error' | 'warning'
  timestamp: string
  checks: Record<string, unknown>
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  source?: string
  timestamp: number
}
