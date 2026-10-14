import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import LoginForm, { LOGIN_ERRORS } from "@/components/auth/LoginForm";

const mockReplace = jest.fn();
let mockSearch = "";

jest.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
  useSearchParams: () => new URLSearchParams(mockSearch),
}));

const fetchMock = jest.fn();

function submit(code: string) {
  fireEvent.change(screen.getByLabelText("Access code"), {
    target: { value: code },
  });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
}

beforeEach(() => {
  mockReplace.mockReset();
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
  mockSearch = "";
});

describe("LoginForm", () => {
  it("focuses the access code field on load", () => {
    render(<LoginForm />);
    expect(screen.getByLabelText("Access code")).toHaveFocus();
    expect(screen.getByLabelText("Access code")).toHaveAttribute(
      "type",
      "password",
    );
  });

  it("posts the code and goes to a safe next path on success", async () => {
    mockSearch = "next=%2Fchat%3Fx%3D1";
    fetchMock.mockResolvedValue({ ok: true, status: 200 });
    render(<LoginForm />);

    submit("correct-horse-battery");

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/chat?x=1"));
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      code: "correct-horse-battery",
    });
  });

  it("ignores a next path that leaves the site", async () => {
    mockSearch = "next=%2F%2Fevil.example";
    fetchMock.mockResolvedValue({ ok: true, status: 200 });
    render(<LoginForm />);

    submit("correct-horse-battery");

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/"));
  });

  it.each([
    [401, LOGIN_ERRORS.wrong],
    [429, LOGIN_ERRORS.tooMany],
    [503, LOGIN_ERRORS.config],
  ])("shows the error for status %p", async (status, message) => {
    fetchMock.mockResolvedValue({ ok: false, status });
    render(<LoginForm />);

    submit("some-code");

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("says sign-in is not set up when the middleware sent ?error=config", () => {
    mockSearch = "error=config";
    render(<LoginForm />);
    expect(screen.getByRole("alert")).toHaveTextContent(LOGIN_ERRORS.config);
  });
});
