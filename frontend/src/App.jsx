import { useEffect, useRef, useState } from 'react'
import {
  Bot,
  ChevronDown,
  ChevronUp,
  FileText,
  Film,
  Loader2,
  Send,
  Settings,
  Sparkles,
  Trash2,
  UploadCloud,
  User,
} from 'lucide-react'

const ACCEPT_ATTR = '.mp4,.mov,.mkv,.avi,.webm,.m4v,.mp3,.wav,.m4a,.ogg,.flac,.pdf,.txt,.md,.json'

export default function App() {
  const [health, setHealth] = useState(null)
  const [ollamaHost, setOllamaHost] = useState('http://localhost:11434')
  const [embedModel, setEmbedModel] = useState('bge-m3')
  const [chatModel, setChatModel] = useState('llama3.2')
  const [whisperModel, setWhisperModel] = useState('base')
  const [language, setLanguage] = useState('auto')
  const [topK, setTopK] = useState(5)
  const [temperature, setTemperature] = useState(0.2)
  const [retrievalOnly, setRetrievalOnly] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const [documents, setDocuments] = useState([])
  const [selectedDocIds, setSelectedDocIds] = useState(new Set())
  const [isDragging, setIsDragging] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [isUploading, setIsUploading] = useState(false)

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [openSources, setOpenSources] = useState({})

  const fileInputRef = useRef(null)
  const messagesEndRef = useRef(null)

  const fetchStatus = async (hostOverride) => {
    try {
      const hostParam = encodeURIComponent(hostOverride || ollamaHost)
      const [hRes, dRes] = await Promise.all([
        fetch(`/api/health?ollama_host=${hostParam}`),
        fetch('/api/documents'),
      ])
      if (hRes.ok) {
        const hData = await hRes.json()
        setHealth(hData)
        if (hData.models?.length > 0) {
          setChatModel((prev) => (hData.models.includes(prev) ? prev : hData.defaults?.chat_model || hData.models[0]))
        }
        if (hData.embed_models?.length > 0) {
          setEmbedModel((prev) =>
            hData.embed_models.includes(prev) ? prev : hData.defaults?.embed_model || hData.embed_models[0],
          )
        }
      }
      if (dRes.ok) {
        const dData = await dRes.json()
        const docs = dData.documents || []
        setDocuments(docs)
        setSelectedDocIds(new Set(docs.map((d) => d.doc_id)))
      }
    } catch {
      setHealth({ ollama_healthy: false, ollama_error: 'Backend API not reachable' })
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleFiles = async (fileList) => {
    const files = Array.from(fileList || [])
    if (!files.length) return

    setUploadError('')
    setIsUploading(true)

    for (const file of files) {
      setUploadStatus(`Processing & indexing "${file.name}"...`)
      const formData = new FormData()
      formData.append('file', file)
      formData.append('embed_model', embedModel)
      formData.append('whisper_model', whisperModel)
      formData.append('language', language)
      formData.append('ollama_host', ollamaHost)

      try {
        const res = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData,
        })
        const data = await res.json()
        if (!res.ok) {
          throw new Error(data.detail || `Failed to upload ${file.name}`)
        }
        await fetchStatus()
      } catch (err) {
        setUploadError(err.message)
      }
    }

    setUploadStatus('')
    setIsUploading(false)
  }

  const handleImportSample = async () => {
    setUploadError('')
    setIsUploading(true)
    setUploadStatus('Loading sample pre-indexed lecture...')
    try {
      const res = await fetch('/api/documents/import-sample', { method: 'POST' })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Could not import sample')
      }
      await fetchStatus()
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploadStatus('')
      setIsUploading(false)
    }
  }

  const handleDeleteDoc = async (docId) => {
    await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' })
    await fetchStatus()
  }

  const handleClearAll = async () => {
    await fetch('/api/documents', { method: 'DELETE' })
    setMessages([])
    await fetchStatus()
  }

  const toggleDocSelection = (docId) => {
    setSelectedDocIds((prev) => {
      const next = new Set(prev)
      if (next.has(docId)) next.delete(docId)
      else next.add(docId)
      return next
    })
  }

  const sendMessage = async () => {
    const query = input.trim()
    if (!query || isStreaming) return
    if (documents.length === 0) {
      setUploadError('Please upload a Video or PDF first.')
      return
    }

    const userMsg = { role: 'user', content: query }
    const historySnapshot = messages.map((m) => ({ role: m.role, content: m.content }))
    const assistantIdx = messages.length + 1

    setMessages((prev) => [...prev, userMsg, { role: 'assistant', content: '', sources: [] }])
    setInput('')
    setIsStreaming(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          chat_history: historySnapshot,
          doc_ids: Array.from(selectedDocIds),
          top_k: topK,
          temperature,
          embed_model: embedModel,
          chat_model: chatModel,
          ollama_host: ollamaHost,
          retrieval_only: retrievalOnly,
        }),
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Chat request failed')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const block of parts) {
          let eventType = 'message'
          let dataStr = ''
          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) eventType = line.slice(7).trim()
            else if (line.startsWith('data: ')) dataStr += line.slice(6)
          }
          if (!dataStr) continue
          const payload = JSON.parse(dataStr)

          if (eventType === 'sources') {
            setMessages((prev) =>
              prev.map((m, i) => (i === assistantIdx ? { ...m, sources: payload } : m)),
            )
          } else if (eventType === 'token') {
            setMessages((prev) =>
              prev.map((m, i) =>
                i === assistantIdx ? { ...m, content: m.content + payload } : m,
              ),
            )
          } else if (eventType === 'error') {
            throw new Error(payload.message || 'Streaming error')
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m, i) =>
          i === assistantIdx ? { ...m, content: `Error: ${err.message}` } : m,
        ),
      )
    } finally {
      setIsStreaming(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const models = health?.models?.length ? health.models : [chatModel]
  const embedOptions = health?.embed_models?.length ? health.embed_models : [embedModel]

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={18} />
          </div>
          <span>Video & PDF RAG</span>
        </div>

        <div className="status-pill">
          <span className={`status-dot ${health?.ollama_healthy ? 'online' : 'offline'}`} />
          <span>
            {health?.ollama_healthy
              ? `Ollama Ready (${health.models.length} models)`
              : 'Ollama Not Reachable'}
          </span>
        </div>

        <div>
          <div className="section-label">Upload Video, Audio, or PDF</div>
          <div
            className={`dropzone ${isDragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setIsDragging(false)
              handleFiles(e.dataTransfer.files)
            }}
          >
            <UploadCloud size={24} color="#818cf8" />
            <p>{isUploading ? 'Processing...' : 'Drop files or click to upload'}</p>
            <span>MP4, MOV, MP3, WAV, PDF, TXT, JSON</span>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={ACCEPT_ATTR}
              style={{ display: 'none' }}
              onChange={(e) => {
                handleFiles(e.target.files)
                e.target.value = ''
              }}
            />
          </div>

          {uploadStatus && (
            <div className="upload-status">
              <Loader2 size={14} className="spin" />
              <span>{uploadStatus}</span>
            </div>
          )}
          {uploadError && <div className="error-banner" style={{ marginTop: 8 }}>{uploadError}</div>}
        </div>

        <div>
          <div className="section-label">
            Knowledge Base ({documents.length} {documents.length === 1 ? 'file' : 'files'})
          </div>
          {documents.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                No files indexed yet.
              </span>
              {health?.store?.has_sample && (
                <button className="btn" onClick={handleImportSample} disabled={isUploading}>
                  Load Sample Lecture
                </button>
              )}
            </div>
          ) : (
            <>
              <div className="doc-list">
                {documents.map((doc) => (
                  <div className="doc-item" key={doc.doc_id}>
                    <div className="doc-item-left">
                      <input
                        type="checkbox"
                        checked={selectedDocIds.has(doc.doc_id)}
                        onChange={() => toggleDocSelection(doc.doc_id)}
                      />
                      {doc.source_type === 'video' || doc.source_type === 'audio' ? (
                        <Film size={15} color="#818cf8" style={{ flexShrink: 0 }} />
                      ) : (
                        <FileText size={15} color="#34d399" style={{ flexShrink: 0 }} />
                      )}
                      <div style={{ minWidth: 0 }}>
                        <div className="doc-name" title={doc.source_name}>
                          {doc.source_name}
                        </div>
                        <div className="doc-meta">
                          {doc.source_type.toUpperCase()} · {doc.chunks} chunks
                        </div>
                      </div>
                    </div>
                    <button
                      className="icon-btn danger"
                      title="Remove document"
                      onClick={() => handleDeleteDoc(doc.doc_id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
              <div className="btn-row" style={{ marginTop: 10 }}>
                <button className="btn" onClick={() => setMessages([])}>
                  Clear Chat
                </button>
                <button className="btn" onClick={handleClearAll}>
                  Reset Index
                </button>
              </div>
            </>
          )}
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button
            className="btn"
            style={{ width: '100%', justifyContent: 'space-between' }}
            onClick={() => setShowSettings((s) => !s)}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Settings size={14} /> Model & RAG Settings
            </span>
            {showSettings ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showSettings && (
            <div style={{ marginTop: 12 }}>
              <div className="form-group">
                <label>Chat Model</label>
                <select
                  className="form-control"
                  value={chatModel}
                  onChange={(e) => setChatModel(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Embedding Model</label>
                <select
                  className="form-control"
                  value={embedModel}
                  onChange={(e) => setEmbedModel(e.target.value)}
                >
                  {embedOptions.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Whisper Model (Video/Audio)</label>
                <select
                  className="form-control"
                  value={whisperModel}
                  onChange={(e) => setWhisperModel(e.target.value)}
                >
                  {['tiny', 'base', 'small', 'medium', 'large-v3'].map((w) => (
                    <option key={w} value={w}>
                      {w}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Top-K Chunks ({topK})</label>
                <input
                  type="range"
                  min={1}
                  max={12}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label>Ollama Host</label>
                <input
                  className="form-control"
                  value={ollamaHost}
                  onChange={(e) => setOllamaHost(e.target.value)}
                  onBlur={() => fetchStatus(ollamaHost)}
                />
              </div>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={retrievalOnly}
                  onChange={(e) => setRetrievalOnly(e.target.checked)}
                />
                Retrieval-only mode (no LLM)
              </label>
            </div>
          )}
        </div>
      </aside>

      <main className="chat-area">
        <header className="chat-header">
          <div>
            <h1>Video & PDF RAG Chatbot</h1>
            <p>
              Real-time local transcription, PDF chunking, and grounded streaming answers with
              timestamps & page citations
            </p>
          </div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Model: {chatModel}
          </span>
        </header>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <Bot size={40} color="#818cf8" />
              <h2>Ask anything about your Videos or PDFs</h2>
              <p>
                Upload a video lecture, audio recording, or PDF document in the sidebar. Ask
                questions and get instant answers cited with exact timestamps or page numbers.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message-row ${msg.role}`}>
                {msg.role === 'assistant' && (
                  <div className="avatar assistant">
                    <Bot size={16} />
                  </div>
                )}
                <div className="bubble">
                  {msg.content || (isStreaming && idx === messages.length - 1 ? 'Thinking...' : '')}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-toggle">
                      <button
                        className="sources-btn"
                        onClick={() =>
                          setOpenSources((prev) => ({ ...prev, [idx]: !prev[idx] }))
                        }
                      >
                        {openSources[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        {msg.sources.length} retrieved sources
                      </button>
                      {openSources[idx] && (
                        <div className="sources-list">
                          {msg.sources.map((s, sIdx) => (
                            <div className="source-card" key={sIdx}>
                              <div className="source-card-header">
                                <strong>{s.source_name}</strong>
                                <span>
                                  {s.location} · Score {s.score.toFixed(3)}
                                </span>
                              </div>
                              <p>{s.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="avatar user">
                    <User size={16} />
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer-wrapper">
          <div className="composer">
            <textarea
              rows={1}
              placeholder="Ask a question about your uploaded videos or PDFs..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
            />
            <button
              className="btn btn-primary"
              onClick={sendMessage}
              disabled={isStreaming || !input.trim()}
            >
              {isStreaming ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

