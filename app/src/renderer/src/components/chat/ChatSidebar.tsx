import { useState } from 'react'
import type { ChatMeta } from '../../lib/chat'

interface ChatSidebarProps {
  chats: ChatMeta[]
  activeChatId: string | null
  streamingChatIds: Set<string>
  onSelect: (chatId: string) => void
  onNewChat: () => void
  onRename: (chatId: string, title: string) => void
  onDelete: (chatId: string) => void
}

export function ChatSidebar({
  chats,
  activeChatId,
  streamingChatIds,
  onSelect,
  onNewChat,
  onRename,
  onDelete
}: ChatSidebarProps): React.JSX.Element {
  return (
    <aside className="chat-sidebar" aria-label="Chats">
      <button className="new-chat-button" type="button" onClick={onNewChat}>
        + New chat
      </button>
      <nav className="chat-tabs" role="tablist" aria-label="Chat threads">
        {chats.map((chat) => (
          <ChatTab
            key={chat.chat_id}
            chat={chat}
            active={chat.chat_id === activeChatId}
            streaming={streamingChatIds.has(chat.chat_id)}
            onSelect={() => onSelect(chat.chat_id)}
            onRename={(title) => onRename(chat.chat_id, title)}
            onDelete={() => onDelete(chat.chat_id)}
          />
        ))}
        {chats.length === 0 ? <div className="chat-tabs-empty">No chats yet</div> : null}
      </nav>
    </aside>
  )
}

interface ChatTabProps {
  chat: ChatMeta
  active: boolean
  streaming: boolean
  onSelect: () => void
  onRename: (title: string) => void
  onDelete: () => void
}

function ChatTab({
  chat,
  active,
  streaming,
  onSelect,
  onRename,
  onDelete
}: ChatTabProps): React.JSX.Element {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(chat.title)
  const [confirmingDelete, setConfirmingDelete] = useState(false)

  const commitRename = (): void => {
    setEditing(false)
    const title = draft.trim()
    if (title && title !== chat.title) {
      onRename(title)
    } else {
      setDraft(chat.title)
    }
  }

  return (
    <div
      className={`chat-tab${active ? ' active' : ''}`}
      role="tab"
      aria-selected={active}
      onClick={onSelect}
      onMouseLeave={() => setConfirmingDelete(false)}
    >
      {editing ? (
        <input
          className="chat-tab-rename"
          value={draft}
          autoFocus
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              commitRename()
            }
            if (event.key === 'Escape') {
              setEditing(false)
              setDraft(chat.title)
            }
          }}
          onBlur={commitRename}
        />
      ) : (
        <>
          <div className="chat-tab-main">
            <span className="chat-tab-title">
              {streaming ? <span className="streaming-dot" aria-label="streaming" /> : null}
              {chat.title || 'New chat'}
            </span>
            <span className="chat-tab-time">{relativeTime(chat.updated)}</span>
          </div>
          <div className="chat-tab-actions" onClick={(event) => event.stopPropagation()}>
            {confirmingDelete ? (
              <button className="chat-tab-action danger" type="button" onClick={onDelete}>
                Delete?
              </button>
            ) : (
              <>
                <button
                  className="chat-tab-action"
                  type="button"
                  aria-label={`Rename ${chat.title || 'chat'}`}
                  onClick={() => {
                    setDraft(chat.title)
                    setEditing(true)
                  }}
                >
                  ✎
                </button>
                <button
                  className="chat-tab-action"
                  type="button"
                  aria-label={`Delete ${chat.title || 'chat'}`}
                  onClick={() => setConfirmingDelete(true)}
                >
                  ×
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function relativeTime(unixSeconds: number): string {
  const deltaS = Math.max(0, Date.now() / 1000 - unixSeconds)
  if (deltaS < 60) {
    return 'now'
  }
  if (deltaS < 3600) {
    return `${Math.floor(deltaS / 60)}m`
  }
  if (deltaS < 86400) {
    return `${Math.floor(deltaS / 3600)}h`
  }
  if (deltaS < 7 * 86400) {
    return `${Math.floor(deltaS / 86400)}d`
  }
  return new Date(unixSeconds * 1000).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric'
  })
}
