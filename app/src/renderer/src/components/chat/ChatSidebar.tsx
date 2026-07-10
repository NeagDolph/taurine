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
    <aside
      className="flex min-h-0 flex-col gap-2 border-r border-white/15 py-2.5 pl-0.5 pr-2.5"
      aria-label="Chats"
    >
      <button
        className="cursor-pointer rounded-[10px] border border-white/10 bg-accent/15 px-3 py-2 text-left text-xs font-semibold text-accent-hi transition-colors hover:bg-accent/25"
        type="button"
        onClick={onNewChat}
      >
        + New chat
      </button>
      <nav
        className="flex min-h-0 flex-col gap-0.5 overflow-y-auto"
        role="tablist"
        aria-label="Chat threads"
      >
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
        {chats.length === 0 ? (
          <div className="px-2.5 py-3 text-xs text-fg/65">No chats yet</div>
        ) : null}
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
      className={`group flex min-h-[38px] cursor-pointer items-center justify-between gap-1.5 rounded-md border-l-2 py-1.5 pl-2 pr-1.5 ${
        active ? 'border-l-accent bg-white/[0.07]' : 'border-l-transparent hover:bg-white/[0.045]'
      }`}
      role="tab"
      aria-selected={active}
      onClick={onSelect}
      onMouseLeave={() => setConfirmingDelete(false)}
    >
      {editing ? (
        <input
          className="w-full rounded-md border border-accent bg-surface-2 px-2 py-1.5 text-xs text-fg outline-none"
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
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="flex items-center gap-1.5 truncate text-xs font-medium text-fg">
              {streaming ? (
                <span
                  className="h-[7px] w-[7px] flex-none animate-pulse rounded-full bg-accent"
                  aria-label="streaming"
                />
              ) : null}
              {chat.title || 'New chat'}
            </span>
            <span className="text-[10px] text-fg/60">{relativeTime(chat.updated)}</span>
          </div>
          <div className="hidden gap-0.5 group-hover:flex" onClick={(event) => event.stopPropagation()}>
            {confirmingDelete ? (
              <button className={`${TAB_ACTION} text-[11px] font-bold !text-bad`} type="button" onClick={onDelete}>
                Delete?
              </button>
            ) : (
              <>
                <button
                  className={TAB_ACTION}
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
                  className={TAB_ACTION}
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

const TAB_ACTION =
  'cursor-pointer rounded border-0 bg-transparent px-1.5 py-1 text-xs text-fg/65 transition-colors hover:bg-white/[0.08] hover:text-fg'

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
