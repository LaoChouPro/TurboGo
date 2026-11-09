import AppKit
import Foundation
import SwiftUI

struct EngineState: Codable {
    let ok: Bool
    let error: String?
    let board: [[Int]]?
    let to_play: Int?
    let ply: Int?
    let last_move: Int?
    let last_coord: String?
    let game_over: Bool?
    let black_winrate: Double?
    let black_score_lead: Double?
}

@MainActor
final class MinimalGoViewModel: ObservableObject {
    @Published var board: [[Int]] = Array(repeating: Array(repeating: 0, count: 19), count: 19)
    @Published var toPlay: Int = 1
    @Published var ply: Int = 0
    @Published var lastMove: Int = -1
    @Published var lastCoord: String = "--"
    @Published var blackWinrate: Double = 50.0
    @Published var blackScoreLead: Double = 0.0
    @Published var gameOver: Bool = false
    @Published var humanColor: Int = 1 // 1 = Black, 2 = White
    @Published var isThinking: Bool = false
    @Published var status: String = "执黑先行"
    @Published var hoverPos: (row: Int, col: Int)? = nil

    private let baseURL = URL(string: "http://127.0.0.1:8788")!

    func connect() {
        Task {
            if let st = try? await request(path: "/api/state", method: "GET") {
                apply(st)
                status = "就绪 · 轮到你落子"
            } else {
                status = "未连接引擎 (127.0.0.1:8788)"
            }
        }
    }

    func click(row: Int, col: Int) {
        guard !isThinking, !gameOver, toPlay == humanColor, board[row][col] == 0 else { return }
        play(move: row * 19 + col)
    }

    func pass() {
        guard !isThinking, !gameOver, toPlay == humanColor else { return }
        play(move: 361)
    }

    private func play(move: Int) {
        isThinking = true
        status = "落子中..."
        Task {
            do {
                let st = try await request(path: "/api/play", body: ["move": move])
                guard st.ok else {
                    status = st.error ?? "禁着点"
                    isThinking = false
                    return
                }
                apply(st)
                if gameOver {
                    status = "对局结束"
                    isThinking = false
                    return
                }
                await aiReply()
            } catch {
                status = "请求失败"
                isThinking = false
            }
        }
    }

    private func aiReply() async {
        status = "AI 思考中..."
        do {
            let st = try await request(path: "/api/ai_move", body: ["top_k": 8])
            apply(st)
            let leadStr = String(format: "%+.1f", blackScoreLead)
            status = gameOver ? "对局结束" : "AI: \(lastCoord) (黑\(Int(round(blackWinrate)))% \(leadStr)目)"
        } catch {
            status = "AI 响应失败"
        }
        isThinking = false
    }

    func newGame(color: Int) {
        guard !isThinking else { return }
        humanColor = color
        isThinking = true
        Task {
            if let st = try? await request(path: "/api/new_game", body: ["komi": 7.5, "top_k": 8]) {
                apply(st)
                if humanColor == 2 {
                    await aiReply()
                } else {
                    status = "新局 · 你执黑"
                    isThinking = false
                }
            } else {
                isThinking = false
            }
        }
    }

    func undo() {
        guard !isThinking, ply > 0 else { return }
        isThinking = true
        let steps = (toPlay == humanColor && ply >= 2) ? 2 : 1
        Task {
            if let st = try? await request(path: "/api/undo", body: ["steps": steps]) {
                apply(st)
                status = "已悔棋 · 第 \(ply) 手"
            }
            isThinking = false
        }
    }

    private func apply(_ st: EngineState) {
        if let b = st.board { board = b }
        if let tp = st.to_play { toPlay = tp }
        if let p = st.ply { ply = p }
        if let lm = st.last_move { lastMove = lm }
        if let lc = st.last_coord { lastCoord = lc }
        if let go = st.game_over { gameOver = go }
        if let wr = st.black_winrate { blackWinrate = wr }
        if let ld = st.black_score_lead { blackScoreLead = ld }
    }

    private func request(path: String, method: String = "POST", body: [String: Any]? = nil) async throws -> EngineState {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        req.timeoutInterval = 30
        if let body = body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONDecoder().decode(EngineState.self, from: data)
    }
}

struct MinimalBoardView: View {
    @ObservedObject var vm: MinimalGoViewModel
    private let cols = Array("ABCDEFGHJKLMNOPQRST")
    private let stars = [
        (3, 3), (3, 9), (3, 15),
        (9, 3), (9, 9), (9, 15),
        (15, 3), (15, 9), (15, 15)
    ]

    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            let pad = side * 0.055
            let span = side - pad * 2.0
            let cell = span / 18.0

            Canvas { ctx, _ in
                // Pure white background
                ctx.fill(Path(CGRect(x: 0, y: 0, width: side, height: side)), with: .color(.white))

                // 19x19 Black grid lines
                for i in 0..<19 {
                    let p = pad + CGFloat(i) * cell
                    let lw: CGFloat = (i == 0 || i == 18) ? 1.5 : 0.8

                    var h = Path()
                    h.move(to: CGPoint(x: pad, y: p))
                    h.addLine(to: CGPoint(x: pad + span, y: p))
                    ctx.stroke(h, with: .color(.black), lineWidth: lw)

                    var v = Path()
                    v.move(to: CGPoint(x: p, y: pad))
                    v.addLine(to: CGPoint(x: p, y: pad + span))
                    ctx.stroke(v, with: .color(.black), lineWidth: lw)
                }

                // Minimalist coordinates
                if side >= 300 {
                    let fSize = max(8.0, cell * 0.34)
                    for i in 0..<19 {
                        let pos = pad + CGFloat(i) * cell
                        let cTxt = Text(String(cols[i]))
                            .font(.system(size: fSize, weight: .regular, design: .monospaced))
                            .foregroundColor(.black.opacity(0.6))
                        ctx.draw(cTxt, at: CGPoint(x: pos, y: pad * 0.42))

                        let rTxt = Text("\(19 - i)")
                            .font(.system(size: fSize, weight: .regular, design: .monospaced))
                            .foregroundColor(.black.opacity(0.6))
                        ctx.draw(rTxt, at: CGPoint(x: pad * 0.42, y: pos))
                    }
                }

                // Star points
                for (sr, sc) in stars {
                    let sx = pad + CGFloat(sc) * cell
                    let sy = pad + CGFloat(sr) * cell
                    let r = max(2.2, cell * 0.10)
                    ctx.fill(Path(ellipseIn: CGRect(x: sx - r, y: sy - r, width: r * 2, height: r * 2)), with: .color(.black))
                }

                // Hover preview
                if let hp = vm.hoverPos, !vm.isThinking, !vm.gameOver, vm.toPlay == vm.humanColor, vm.board[hp.row][hp.col] == 0 {
                    let cx = pad + CGFloat(hp.col) * cell
                    let cy = pad + CGFloat(hp.row) * cell
                    let rad = cell * 0.44
                    let rect = CGRect(x: cx - rad, y: cy - rad, width: rad * 2, height: rad * 2)
                    if vm.toPlay == 1 {
                        ctx.fill(Path(ellipseIn: rect), with: .color(.black.opacity(0.28)))
                    } else {
                        ctx.fill(Path(ellipseIn: rect), with: .color(.white.opacity(0.85)))
                        ctx.stroke(Path(ellipseIn: rect), with: .color(.black.opacity(0.45)), lineWidth: 1.2)
                    }
                }

                // Stones (pure flat Black & White)
                for r in 0..<19 {
                    for c in 0..<19 {
                        let s = vm.board[r][c]
                        guard s != 0 else { continue }
                        let cx = pad + CGFloat(c) * cell
                        let cy = pad + CGFloat(r) * cell
                        let rad = cell * 0.455
                        let rect = CGRect(x: cx - rad, y: cy - rad, width: rad * 2, height: rad * 2)

                        if s == 1 {
                            ctx.fill(Path(ellipseIn: rect), with: .color(.black))
                        } else {
                            ctx.fill(Path(ellipseIn: rect), with: .color(.white))
                            ctx.stroke(Path(ellipseIn: rect), with: .color(.black), lineWidth: max(1.1, cell * 0.045))
                        }

                        // Last move marker (minimalist contrasting ring)
                        if vm.lastMove == r * 19 + c {
                            let mr = rad * 0.38
                            let mRect = CGRect(x: cx - mr, y: cy - mr, width: mr * 2, height: mr * 2)
                            ctx.stroke(
                                Path(ellipseIn: mRect),
                                with: .color(s == 1 ? .white : .black),
                                lineWidth: max(1.3, cell * 0.055)
                            )
                        }
                    }
                }
            }
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onEnded { v in
                        let col = Int(round((v.location.x - pad) / cell))
                        let row = Int(round((v.location.y - pad) / cell))
                        if (0..<19).contains(row) && (0..<19).contains(col) {
                            vm.click(row: row, col: col)
                        }
                    }
            )
            .onContinuousHover { phase in
                switch phase {
                case .active(let loc):
                    let col = Int(round((loc.x - pad) / cell))
                    let row = Int(round((loc.y - pad) / cell))
                    if (0..<19).contains(row) && (0..<19).contains(col) {
                        vm.hoverPos = (row, col)
                    } else {
                        vm.hoverPos = nil
                    }
                case .ended:
                    vm.hoverPos = nil
                }
            }
            .frame(width: side, height: side)
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .background(Color.white)
    }
}

struct RootView: View {
    @StateObject private var vm = MinimalGoViewModel()

    var body: some View {
        VStack(spacing: 0) {
            MinimalBoardView(vm: vm)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            // Ultra-minimalist monochrome bottom bar
            HStack(spacing: 12) {
                Text(vm.status)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.black)
                    .lineLimit(1)

                Spacer()

                Button("执黑") { vm.newGame(color: 1) }
                Button("执白") { vm.newGame(color: 2) }
                Button("悔棋") { vm.undo() }
                    .disabled(vm.ply == 0 || vm.isThinking)
                Button("虚手") { vm.pass() }
                    .disabled(vm.isThinking || vm.gameOver)
            }
            .buttonStyle(.plain)
            .font(.system(size: 12, weight: .medium, design: .monospaced))
            .foregroundColor(.black)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(white: 0.96))
        }
        .background(Color.white)
        .onAppear { vm.connect() }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@main
struct MinimalGoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup("TurboGo-v2") {
            RootView()
                .frame(minWidth: 260, minHeight: 280)
                .preferredColorScheme(.light)
        }
        .defaultSize(width: 540, height: 570)
    }
}
