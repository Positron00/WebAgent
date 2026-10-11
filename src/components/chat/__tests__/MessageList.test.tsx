import { render, screen } from '@/test-utils';
import { MessageList } from '../MessageList';
import { ChatMessage } from '@/types/chat';

describe('MessageList Component', () => {
  const mockMessages: ChatMessage[] = [
    { role: 'user', content: 'Hello' },
    { role: 'assistant', content: 'Hi there!' },
  ];

  it('renders empty state when no messages', () => {
    render(<MessageList messages={[]} isLoading={false} />);
    
    expect(screen.getByText('Start a conversation...')).toBeInTheDocument();
    expect(screen.getByText('Responses will appear here')).toBeInTheDocument();
  });

  it('renders messages correctly', () => {
    render(<MessageList messages={mockMessages} isLoading={false} />);
    
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    render(<MessageList messages={mockMessages} isLoading={true} />);
    
    expect(screen.getByText('Thinking...')).toBeInTheDocument();
  });

  it('shows error message', () => {
    const error = 'Something went wrong';
    render(<MessageList messages={mockMessages} isLoading={false} error={error} />);
    
    expect(screen.getByText(error)).toBeInTheDocument();
  });

  it('applies correct roles to messages', () => {
    render(<MessageList messages={mockMessages} isLoading={false} />);
    
    expect(screen.getByRole('note')).toHaveTextContent('Hello');
    expect(screen.getByRole('article')).toHaveTextContent('Hi there!');
  });

  it('shows a second answer to the same question (Research this in depth)', () => {
    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', content: 'What is a tuple?' },
      { id: 'a1', role: 'assistant', content: 'Short answer',
        route: { route: 'direct', decidedBy: 'model', reason: 'Auto · definition' } },
      { id: 'a2', role: 'assistant', content: 'Researched answer',
        route: { route: 'agentic', decidedBy: 'override', reason: 'You asked for research' } },
    ];
    render(<MessageList messages={messages} isLoading={false} />);

    expect(screen.getByText('Short answer')).toBeInTheDocument();
    expect(screen.getByText('Researched answer')).toBeInTheDocument();
    expect(screen.getByText('Answered directly')).toBeInTheDocument();
    expect(screen.getByText('Deep research')).toBeInTheDocument();
  });
});

